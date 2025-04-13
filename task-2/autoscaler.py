import asyncio
import httpx
import subprocess
import time
import os
import logging

# Configuration constants
CHECK_INTERVAL = 5  # Interval for checking system status
MAX_INSTANCES = 3
MIN_INSTANCES = 2
SCALE_UP_THRESHOLD = 29.0  # RPS to scale up
SCALE_DOWN_THRESHOLD = 25.0  # RPS to scale down
SCALE_UP_COOLDOWN = 30  # Cooldown before scaling up again
SCALE_DOWN_COOLDOWN = 120  # Cooldown before scaling down again
BACKEND_BASE_PORT = 8000
SERVING_SCRIPT = "serving_rag_batched.py"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for GPU configuration
os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"

processes = {}

async def register_with_balancer(port: int):
    """Registers an instance with the load balancer."""
    url = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        try:
            await client.post("http://localhost:9000/register", json={"url": url})
            logger.info(f"Registered instance on port {port}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to register instance on port {port}: {e}")

async def unregister_from_balancer(port: int):
    """Unregisters an instance from the load balancer."""
    url = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        try:
            await client.post("http://localhost:9000/unregister", json={"url": url})
            logger.info(f"Unregistered instance on port {port}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to unregister instance on port {port}: {e}")

async def start_instance(port: int):
    """Starts a new backend instance if not running."""
    if port in processes:
        logger.info(f"Instance on port {port} is already running.")
        return

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    try:
        proc = subprocess.Popen(["python", SERVING_SCRIPT], env=env)
        processes[port] = proc
        logger.info(f"Started instance on port {port}")

        if await wait_until_ready(port):
            logger.info(f"Instance on port {port} is ready.")
        else:
            logger.warning(f"Timeout: Instance on port {port} did not become ready.")
    except Exception as e:
        logger.error(f"Failed to start instance on port {port}: {e}")

async def stop_instance(port: int):
    """Stops a running backend instance."""
    proc = processes.get(port)
    if proc:
        proc.terminate()
        proc.wait()
        del processes[port]
        logger.info(f"Stopped instance on port {port}")
        await unregister_from_balancer(port)

async def get_total_rps() -> float:
    """Fetches the total RPS from the load balancer."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:9000/metrics")
            resp.raise_for_status()
            data = resp.json()
            total_rps = sum(data["rps"].values())
            logger.info(f"Total RPS: {total_rps}")
            return total_rps
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.error(f"Error fetching RPS: {e}")
        return 0.0

async def autoscaler_loop():
    """Main loop for autoscaling backend instances."""
    current_instances = MIN_INSTANCES
    await scale_up_instances(current_instances)

    last_scale_up_time = 0
    last_scale_down_time = 0

    while True:
        total_rps = await get_total_rps()

        now = time.time()
        can_scale_up = now - last_scale_up_time >= SCALE_UP_COOLDOWN
        can_scale_down = now - last_scale_down_time >= SCALE_DOWN_COOLDOWN

        # Scale up if conditions are met
        if total_rps >= SCALE_UP_THRESHOLD and current_instances < MAX_INSTANCES and can_scale_up:
            port = BACKEND_BASE_PORT + current_instances
            await start_instance(port)
            await asyncio.sleep(15)  # Wait for model to load
            await register_with_balancer(port)
            current_instances += 1
            last_scale_up_time = now

        # Scale down if conditions are met
        elif total_rps < SCALE_DOWN_THRESHOLD and current_instances > MIN_INSTANCES and can_scale_down:
            port = BACKEND_BASE_PORT + current_instances - 1
            await stop_instance(port)
            current_instances -= 1
            last_scale_down_time = now

        await asyncio.sleep(CHECK_INTERVAL)

async def wait_until_ready(port: int, timeout: int = 200) -> bool:
    """Waits until the instance is ready to handle requests."""
    url = f"http://localhost:{port}/rag"
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = await client.post(url, json={"query": "ping", "k": 1})
                if resp.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(2)
    return False

async def scale_up_instances(current_instances: int):
    """Scales up initial backend instances."""
    for i in range(current_instances):
        port = BACKEND_BASE_PORT + i
        await start_instance(port)
        await register_with_balancer(port)

if __name__ == "__main__":
    asyncio.run(autoscaler_loop())  # Run the autoscaling loop
