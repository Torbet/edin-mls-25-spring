import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from collections import defaultdict
import logging
from contextlib import asynccontextmanager
import uvicorn

BACKENDS: List[str] = []
backend_index = 0
backend_lock = asyncio.Lock()

request_counts: Dict[str, int] = defaultdict(int)
stored_rps: Dict[str, float] = defaultdict(float)
last_reset = time.time()
reset_interval = 5

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def reset_loop():
        global last_reset
        while True:
            await asyncio.sleep(reset_interval)
            now = time.time()
            elapsed = now - last_reset or 1e-6
            last_reset = now
            async with backend_lock:
                for backend in BACKENDS:
                    rps = request_counts[backend] / elapsed
                    stored_rps[backend] = round(rps, 2)
                    request_counts[backend] = 0
    asyncio.create_task(reset_loop())
    yield

app.router.lifespan_context = lifespan

class BackendRequest(BaseModel):
    url: str

@app.post("/register")
async def register_backend(request: BackendRequest):
    async with backend_lock:
        if request.url not in BACKENDS:
            BACKENDS.append(request.url)
            logger.info(f"Registered backend: {request.url}")
        else:
            raise HTTPException(status_code=400, detail="Backend already registered")
    return {"status": "registered", "backend": request.url}

@app.post("/unregister")
async def unregister_backend(request: BackendRequest):
    async with backend_lock:
        if request.url in BACKENDS:
            BACKENDS.remove(request.url)
            logger.info(f"Unregistered backend: {request.url}")
        else:
            raise HTTPException(status_code=400, detail="Backend not found")
    return {"status": "unregistered", "backend": request.url}

@app.get("/assign")
async def assign_backend():
    global backend_index
    async with backend_lock:
        if not BACKENDS:
            raise HTTPException(status_code=503, detail="No backends available")
        backend = BACKENDS[backend_index % len(BACKENDS)]
        backend_index = (backend_index + 1) % len(BACKENDS)
        request_counts[backend] += 1
    logger.info(f"Assigned backend: {backend}/rag")
    return {"backend": f"{backend}/rag"}

@app.get("/metrics")
async def get_metrics():
    return {
        "interval_seconds": reset_interval,
        "rps": stored_rps,
        "backend_list": BACKENDS,
    }

if __name__ == "__main__":
    uvicorn.run("load_balancer:app", host="0.0.0.0", port=9000, reload=False)
