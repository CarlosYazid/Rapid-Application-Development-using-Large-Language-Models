"""Single-GPU course image service. Responses contain shared notebook filesystem paths."""
import asyncio
import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_ID = 'black-forest-labs/FLUX.2-klein-4B'
REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
# Relative paths resolve in both the lab and the separately mounted grader.
IMAGE_DIR = Path(os.getenv('DLI_IMAGE_DIR', 'generated_images'))
logger = logging.getLogger(__name__)


#########################################################################################
## Lifespan: load the pipeline, warm up GPU allocations, and clean up on shutdown
#########################################################################################

@asynccontextmanager
async def lifespan(app):
    import torch
    from diffusers import Flux2KleinPipeline
    if not torch.cuda.is_available():
        raise RuntimeError('FLUX requires a CUDA GPU. Start this service in the course GPU lab.')
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    app.state.pipeline = None
    app.state.lock = asyncio.Lock()
    try:
        pipeline = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        )
        # Offload inactive components so the image model and VLM share one GPU.
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=True)
        # Allocate inference buffers before reporting readiness.
        pipeline(prompt='A blank test image', width=512, height=512, num_inference_steps=4, guidance_scale=1.0,
                 generator=torch.Generator('cuda').manual_seed(0))
        app.state.pipeline = pipeline
        yield
    finally:
        app.state.pipeline = None
        if 'pipeline' in locals():
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


#########################################################################################
## Request schema: define the API and bound the work that fits this shared GPU
#########################################################################################

class ImageRequest(BaseModel):
    model: Literal['black-forest-labs/FLUX.2-klein-4B']
    prompt: str = Field(min_length=1, max_length=4000)
    size: str = '1024x1024'
    n: int = Field(default=1, ge=1, le=4)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    @field_validator('size')
    @classmethod
    def valid_size(cls, value):
        try:
            width, height = map(int, value.lower().split('x'))
        except ValueError as error:
            raise ValueError('Use WIDTHxHEIGHT, for example 512x512.') from error
        if any(side < 512 or side > 1024 or side % 64 for side in (width, height)):
            raise ValueError('Each side must be 512..1024 pixels and divisible by 64.')
        return f'{width}x{height}'


@app.get('/health')
@app.get('/')
async def health():
    ready = getattr(app.state, 'pipeline', None) is not None
    if not ready:
        raise HTTPException(503, 'FLUX is not ready.')
    return {'status': 'ready', 'model': MODEL_ID, 'revision': REVISION}


@app.get('/v1/models')
async def models():
    return {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model'}]}


#########################################################################################
## Inference: generate images, retain their files, and return paths to the caller
#########################################################################################

def render(request):
    import torch
    width, height = map(int, request.size.split('x'))
    paths = []
    try:
        # Generate one image at a time so n does not multiply peak GPU memory.
        for index in range(request.n):
            generator = torch.Generator('cuda')
            generator.manual_seed(request.seed + index) if request.seed is not None else generator.seed()
            image = app.state.pipeline(
                prompt=request.prompt, width=width, height=height, num_inference_steps=4, guidance_scale=1.0,
                generator=generator,
            ).images[0]
            target = IMAGE_DIR / f'draw_{uuid4().hex}.png'
            image.save(target)
            paths.append(target)
        return {'data': [{'url': str(path)} for path in paths]}
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


#########################################################################################
## Endpoint: serialize pipeline access while keeping the event loop responsive
#########################################################################################

@app.post('/v1/images/generations')
async def generate(request: ImageRequest):
    await health()
    try:
        await asyncio.wait_for(app.state.lock.acquire(), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(429, 'Image generation is busy; retry after the current request finishes.')
    task = asyncio.create_task(asyncio.to_thread(render, request))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A running CUDA call cannot be canceled by abandoning its Python thread.
        # Hold the lock until it finishes to prevent concurrent pipeline mutation.
        try:
            await task
        except Exception:
            pass
        raise
    except Exception as error:
        logger.error('Image generation failed (%s)', type(error).__name__)
        raise HTTPException(503, 'Image generation failed. Inspect the FLUX service log.') from None
    finally:
        app.state.lock.release()
