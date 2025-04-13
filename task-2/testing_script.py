import asyncio
import httpx
import time
import random
import argparse
import traceback
from typing import List, Tuple, Optional
from statistics import mean, median, quantiles

# ---------------------------- Constants & Configuration ----------------------------

TIMEOUT = 60.0
QUERY = 'Which animals can hover in the air?'
K = 2
POISSON_SEED = 132

# Define endpoints
ENDPOINTS = {
    'original': 'http://localhost:8000/rag',
    'load_balancer': 'http://localhost:9000/assign'
}

# ---------------------------- Data Classes ----------------------------

class RequestResult:
    """Stores the latency and HTTP status code of a request."""
    def __init__(self, latency: float, status_code: int):
        self.latency = latency
        self.status_code = status_code

# ---------------------------- Core Request Logic ----------------------------

async def send_single_request(client: httpx.AsyncClient, endpoint: str, target: str) -> RequestResult:
    """
    Sends a single HTTP request, either directly or via a load balancer.
    Returns a RequestResult with latency and status.
    """
    payload = {'query': QUERY, 'k': K}
    start = time.time()
    try:
        if target == 'load_balancer':
            # Request backend assignment from load balancer
            assign_resp = await client.get(endpoint)
            assigned_backend = assign_resp.json().get('backend')
            if not assigned_backend:
                raise ValueError('No backend assigned.')
            response = await client.post(assigned_backend, json=payload)
        else:
            # Send directly to original endpoint
            response = await client.post(endpoint, json=payload)

        latency = time.time() - start
        return RequestResult(latency, response.status_code)

    except Exception as e:
        print(f'[ERROR] Request failed: {e}')
        traceback.print_exc()
        return RequestResult(latency=float('inf'), status_code=0)

# ---------------------------- Load Testing Logic ----------------------------

async def load_test(rps: int, num_requests: int, mode: str, endpoint: str, target: str) -> Tuple[List[RequestResult], float]:
    """
    Executes a load test with the given parameters:
    - rps: Requests per second
    - num_requests: Total number of requests
    - mode: 'ideal' or 'poisson' request timing
    - endpoint: Target endpoint URL
    - target: 'original' or 'load_balancer'
    Returns a list of results and the simulated total time taken to send all requests.
    """
    if mode == 'poisson':
        random.seed(POISSON_SEED)

    results = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = []

        for i in range(num_requests):
            task = asyncio.create_task(send_single_request(client, endpoint, target))
            tasks.append(task)

            # Throttle request rate based on selected mode
            if mode == 'ideal':
                await asyncio.sleep(1 / rps)
            elif mode == 'poisson':
                await asyncio.sleep(random.expovariate(rps))
            else:
                raise ValueError("Invalid mode. Choose 'ideal' or 'poisson'.")

        responses = await asyncio.gather(*tasks)
        total_time = sum(r.latency for r in responses if r.status_code == 200)
        return responses, total_time / rps if rps else 0.0

# ---------------------------- Metrics Calculation ----------------------------

def compute_metrics(results: List[RequestResult], wall_time: float) -> dict:
    """
    Computes statistics from the list of RequestResult:
    - success/failure counts
    - average latency
    - 95th percentile latency
    - throughput (successful requests per second)
    """
    successful_latencies = [r.latency for r in results if r.status_code == 200]
    failed_count = len([r for r in results if r.status_code != 200])

    return {
        'requests_sent': len(results),
        'successful': len(successful_latencies),
        'failed': failed_count,
        'average_latency': mean(successful_latencies) if successful_latencies else None,
        'p95_latency': quantiles(successful_latencies, n=20)[18] if len(successful_latencies) >= 20 else None,
        'throughput_rps': len(successful_latencies) / wall_time if wall_time > 0 else None,
    }

# ---------------------------- Main Execution Entry ----------------------------

async def main():
    """
    Parses arguments and runs the full load test.
    """
    parser = argparse.ArgumentParser(description='Async Load Test Runner')
    parser.add_argument('--mode', choices=['ideal', 'poisson'], default='ideal', help='Request timing mode')
    parser.add_argument('--target', choices=list(ENDPOINTS.keys()), default='original', help='Target endpoint')
    parser.add_argument('--rps', type=int, required=True, help='Requests per second')
    parser.add_argument('--num_requests', type=int, required=True, help='Total number of requests')

    args = parser.parse_args()
    endpoint = ENDPOINTS[args.target]

    print(f'Running test with:\n - Mode: {args.mode}\n - Target: {args.target}\n - RPS: {args.rps}\n - Total Requests: {args.num_requests}\n')

    start = time.time()
    results, simulated_time = await load_test(args.rps, args.num_requests, args.mode, endpoint, args.target)
    wall_time = time.time() - start

    metrics = compute_metrics(results, wall_time)
    
    print('\n--- Test Results ---')
    for key, value in metrics.items():
        print(f'{key}: {value}')

# ---------------------------- Script Entrypoint ----------------------------

if __name__ == '__main__':
    asyncio.run(main())
