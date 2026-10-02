"""Own the three Notebook 6.5 processes without blocking the notebook kernel."""
import atexit
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

CONFIG = json.loads(Path(__file__).with_name('local-models.json').read_text())


def gpu_budget():
    """Reserve vision weights/KV cache, diffusion workspace, and CUDA headroom."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits', '-i', '0'],
        capture_output=True, text=True, check=True,
    )
    total, free = (int(value.strip()) / 1024 for value in result.stdout.strip().split(','))
    required = CONFIG['vision']['memory_gib'] + CONFIG['diffusion']['reserve_gib'] + CONFIG['headroom_gib']
    if free < required:
        raise RuntimeError(f'The local models need {required:g} GiB free on GPU 0; {free:.1f} GiB is free. '
                           'Shut down earlier notebook kernels before starting these servers.')
    return {'total_gib': round(total, 1), 'free_gib': round(free, 1), 'required_gib': required,
            'vision_fraction': CONFIG['vision']['memory_gib'] / total}


def read_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        body = response.read()
        return json.loads(body) if body.strip() else {'status': 'ready'}


class CourseServers:
    def __init__(self, directory='.', startup_timeout=900):
        self.directory = Path(directory).resolve()
        self.logs = self.directory / 'server_logs'
        self.timeout = float(startup_timeout)
        if not 0 < self.timeout <= 7200:
            raise ValueError('startup_timeout must be between 0 and 7200 seconds.')
        self.processes = {}
        self.budget = None
        atexit.register(self.stop)

    async def start(self):
        if self.processes:
            raise RuntimeError('This launcher already owns services. Inspect status() or call stop() first.')
        for package in ('vllm', 'diffusers'):
            if version(package) != CONFIG[package + '_version']:
                raise RuntimeError(f'{package} must match the pinned course version {CONFIG[package + "_version"]}.')
        for port in (9002, 9003, 9004):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} is occupied. Stop its existing launcher before starting another.')
        self.budget = await asyncio.to_thread(gpu_budget)
        print(f'GPU: {self.budget["free_gib"]} GiB free; {self.budget["required_gib"]} GiB reserved for this workflow.')
        self.logs.mkdir(exist_ok=True)
        vision = CONFIG['vision']
        commands = [
            ('VLM', [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                     '--model', vision['model'], '--revision', vision['revision'],
                     '--code-revision', vision['code_revision'], '--trust-remote-code',
                     '--max-model-len', str(vision['max_model_len']),
                     '--max-num-seqs', str(vision['max_num_seqs']),
                     '--gpu-memory-utilization', str(self.budget['vision_fraction']),
                     '--limit-mm-per-prompt', json.dumps({'image': vision['images_per_prompt'], 'video': 0}),
                     '--enforce-eager', '--host', '0.0.0.0', '--port', '9002'], 9002),
            ('FLUX', [sys.executable, '-m', 'uvicorn', 'sdxl_app:app', '--host', '0.0.0.0', '--port', '9003'], 9003),
            ('ROUTER', [sys.executable, '-m', 'uvicorn', 'router_app:app', '--host', '0.0.0.0', '--port', '9004'], 9004),
        ]
        try:
            for label, command, port in commands:
                log = self.logs / (label.lower() + '.log')
                with log.open('wb') as output:
                    process = subprocess.Popen(command, cwd=self.directory, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                self.processes[label] = (process, port)
                deadline = time.monotonic() + self.timeout
                print(f'[{label}] Loading; log: {log.relative_to(self.directory)}')
                while True:
                    for name, (owned, _) in self.processes.items():
                        if owned.poll() is not None:
                            raise RuntimeError(f'{name} exited with code {owned.returncode}. Inspect server_logs/{name.lower()}.log.')
                    try:
                        await asyncio.to_thread(read_json, f'http://127.0.0.1:{port}/health')
                        if label == 'VLM':
                            catalog = await asyncio.to_thread(read_json, f'http://127.0.0.1:{port}/v1/models')
                            if vision['model'] not in {row['id'] for row in catalog['data']}:
                                raise RuntimeError('Vision service reports the wrong model.')
                        break
                    except (urllib.error.URLError, TimeoutError, ValueError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f'{label} did not become ready in {self.timeout:g}s. Inspect {log.name}.') from None
                        await asyncio.sleep(1)
                print(f'[{label}] Ready on port {port}')
            return self.status()
        except BaseException:
            await asyncio.to_thread(self.stop)
            raise

    def status(self):
        result = {}
        for label, (process, port) in self.processes.items():
            if process.poll() is not None:
                result[label] = {'status': 'exited', 'exit_code': process.returncode}
                continue
            try:
                result[label] = read_json(f'http://127.0.0.1:{port}/health')
            except (urllib.error.URLError, TimeoutError, ValueError):
                result[label] = {'status': 'unavailable', 'log': f'server_logs/{label.lower()}.log'}
        return result

    def stop(self):
        # Kill only process groups created by this instance, never arbitrary GPU jobs.
        for process, _ in reversed(list(self.processes.values())):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
                # Workers may outlive an exited supervisor.
                os.killpg(process.pid, signal.SIGKILL)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
        self.processes.clear()
