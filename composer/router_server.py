import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import httpx
import json
import os
import traceback
from typing import Dict, Any, List, Tuple
from starlette.background import BackgroundTask

# --- Configuration and Defaults ---

DEFAULTS = {
    "TIMEOUT": 330,  # Request timeout in seconds for backend calls
    "URLS": "127.0.0.1:9002,llm_client:9000",  # Backend URLs
    "FILTER_KEYWORDS": "meta,mistral,nvidia,llama-3.2,huggingfacetb",  # Model filter keywords
    "CACHE_REFRESH_INTERVAL": 3600, # Auto-refresh interval in seconds (1 hour)
}

def get_var(key: str) -> str:
    """Fetch a configuration variable, falling back to defaults."""
    return os.getenv(key, DEFAULTS.get(key, ""))

def get_urls() -> List[str]:
    """Parse and normalize backend URLs."""
    # Ensure all URLs start with 'http://' or 'https://'
    return [url if "://" in url else f"http://{url.strip()}" for url in get_var("URLS").split(",")]

def get_filter_keywords() -> List[str]:
    """Parse filter keywords for model filtering."""
    return [kw.strip().lower() for kw in get_var("FILTER_KEYWORDS").split(",")]

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length', 'content-encoding'}


def response_headers(response):
    return {key: value for key, value in response.headers.items() if key.lower() not in HOP_HEADERS}


# --- Utilities for Model Discovery ---

async def fetch_models_from_server(server_url: str) -> Dict[str, Any]:
    """Fetch available models from a single backend server."""
    timeout_sec = int(get_var("TIMEOUT"))
    try:
        # Use a fresh client for each request
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.get(f"{server_url}/v1/models")
            if response.status_code == 200:
                entries = response.json().get("data", [])
                if not isinstance(entries, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in entries):
                    raise ValueError("Invalid model catalog")
                return {"server": server_url, "models": entries}
            return {"error": f"Failed to fetch models from {server_url} (Status: {response.status_code})"}
    except Exception as e:
        # Log error but don't re-raise, discovery should be fault-tolerant
        return {"error": f"Model discovery failed: {type(e).__name__}"}

async def discover_models_and_cache(app: FastAPI) -> None:
    """Discover models from all backend servers and update the application cache."""
    print("--- Starting model discovery ---")

    tasks = [fetch_models_from_server(server) for server in get_urls()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    aggregated_models, errors = [], []
    for server, result in zip(get_urls(), results):
        if isinstance(result, dict) and "models" in result:
            # Tag models with their originating server
            for model in result["models"]:
                model["server"] = server
                aggregated_models.append(model)
        elif isinstance(result, dict) and "error" in result:
            errors.append(result["error"])
        else:
            errors.append(f"Unexpected result from {server}: {result}")

    # Apply filtering only to the discovered models
    keywords = get_filter_keywords()
    filtered_models = [
        model for model in aggregated_models
        if any(kw in model.get("id", "").lower() for kw in keywords)
    ]

    # Update cache
    app.state.models = filtered_models
    app.state.routes = {row["id"]: row["server"] for row in filtered_models}
    app.state.discovery_errors = errors
    print(f"--- Model discovery complete. Found {len(filtered_models)} models. Errors: {len(errors)} ---")

# --- Application Lifecycle and Caching ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    print(f"Configured backend servers: {get_urls()}")
    print(f"Model filter keywords: {get_filter_keywords()}")

    # Initialize state
    app.state.models: List[Dict[str, Any]] = []
    app.state.discovery_errors: List[str] = []
    app.state.timeout = int(get_var("TIMEOUT"))

    # Initial model discovery
    await discover_models_and_cache(app)

    # Setup periodic refresh (Optional, but good for dynamic backends)
    # Ref: https://fastapi.tiangolo.com/tutorial/events/
    # loop = asyncio.get_event_loop()
    # app.state.refresh_task = loop.create_task(
    #     periodic_model_refresh(app, int(get_var("CACHE_REFRESH_INTERVAL")))
    # )

    async with httpx.AsyncClient(timeout=httpx.Timeout(app.state.timeout, connect=10),
                                 trust_env=False, follow_redirects=False) as client:
        app.state.client = client
        yield # App is running

    # Cleanup on shutdown
    # app.state.refresh_task.cancel()
    print("Application shutdown complete.")

async def periodic_model_refresh(app: FastAPI, interval: int) -> None:
    """Periodically refresh the model cache."""
    while True:
        await asyncio.sleep(interval)
        print("--- Initiating periodic model cache refresh ---")
        await discover_models_and_cache(app)

# Initialize FastAPI application and apply lifespan
app = FastAPI(lifespan=lifespan)

# --- Endpoints ---

@app.get("/v1/models")
@app.get("/v1/models/")
async def list_models():
    """List all available (and filtered) models from the cache."""
    response = {"object": "list", "data": app.state.models}
    if app.state.discovery_errors:
        response["warnings"] = app.state.discovery_errors
    return JSONResponse(content=response)

@app.get("/v1/models/refresh")
async def refresh_models():
    """Trigger a manual refresh of the model cache."""
    await discover_models_and_cache(app)
    return JSONResponse(content={"status": "Model cache refreshed.", "count": len(app.state.models)})

async def refresh():
    await discover_models_and_cache(app)


@app.post('/v1/{operation:path}')
async def proxy(request: Request, operation: str):
    if operation not in {'chat/completions', 'completions', 'embeddings'}:
        raise HTTPException(404, 'Unsupported model operation.')
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, 'Invalid JSON request.')
    if not isinstance(body, dict) or not isinstance(body.get('model'), str):
        raise HTTPException(422, 'A model ID is required.')
    if not isinstance(body.get('stream', False), bool):
        raise HTTPException(422, 'stream must be a boolean.')
    model = body['model']
    if model not in app.state.routes:
        await refresh()
    target = app.state.routes.get(model)
    if target is None:
        raise HTTPException(404, 'Model is unavailable. Refresh /v1/models and inspect service readiness.')
    headers = {key: value for key, value in request.headers.items()
               if key.lower() in {'accept', 'content-type', 'cache-control'}}
    upstream = app.state.client.build_request('POST', target.rstrip('/') + '/v1/' + operation,
                                              json=body, headers=headers)
    try:
        response = await app.state.client.send(upstream, stream=True)
    except httpx.TimeoutException:
        raise HTTPException(504, 'The selected model service timed out.') from None
    except httpx.HTTPError:
        raise HTTPException(502, 'The selected model service could not be reached.') from None
    if not body.get('stream') or response.status_code >= 300:
        try:
            content = await response.aread()
            return Response(content, response.status_code, headers=response_headers(response))
        except httpx.TimeoutException:
            raise HTTPException(504, 'The selected model response timed out.') from None
        except httpx.HTTPError:
            raise HTTPException(502, 'The selected model response was interrupted.') from None
        finally:
            await response.aclose()

    async def chunks():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()

    return StreamingResponse(chunks(), status_code=response.status_code,
                             headers=response_headers(response), background=BackgroundTask(response.aclose))


# --- Health Check ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    required = {"nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "nvidia/nemotron-3.5-lightning-30b-a3b"}
    missing = sorted(required - set(app.state.routes))
    return JSONResponse({"status": "degraded" if missing else "ready", "missing_models": missing,
                         "discovery_errors": app.state.discovery_errors}, status_code=503 if missing else 200)