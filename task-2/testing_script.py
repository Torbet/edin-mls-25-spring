import asyncio
import httpx
import time
import random
import argparse
import traceback
from typing import List, Tuple, Optional
from statistics import mean, median, quantiles

# Constants
TIMEOUT = 60.0
QUERY = 'Which animals can hover in the air?'
K = 2
POISSON_SEED = 42

ENDPOINTS = {'original': 'http://localhost:8000/rag', 'load_balancer': 'http://localhost:9000/assign'}


class RequestResult:
  def __init__(self, latency: float, status_code: int):
    self.latency = latency
    self.status_code = status_code


async def send_single_request(client: httpx.AsyncClient, endpoint: str, target: str) -> RequestResult:
  payload = {'query': QUERY, 'k': K}
  start = time.time()
  try:
    if target == 'load_balancer':
      assign_resp = await client.get(endpoint)
      assigned_backend = assign_resp.json().get('backend')
      if not assigned_backend:
        raise ValueError('No backend assigned.')
      response = await client.post(assigned_backend, json=payload)
    else:
      response = await client.post(endpoint, json=payload)

    latency = time.time() - start
    return RequestResult(latency, response.status_code)
  except Exception as e:
    print(f'[ERROR] Request failed: {e}')
    traceback.print_exc()
    return RequestResult(latency=float('inf'), status_code=0)


async def load_test(rps: int, num_requests: int, mode: str, endpoint: str, target: str) -> Tuple[List[RequestResult], float]:
  if mode == 'poisson':
    random.seed(POISSON_SEED)

  results = []
  async with httpx.AsyncClient(timeout=TIMEOUT) as client:
    tasks = []

    for i in range(num_requests):
      task = asyncio.create_task(send_single_request(client, endpoint, target))
      tasks.append(task)

      # Throttle based on mode
      if mode == 'ideal':
        await asyncio.sleep(1 / rps)
      elif mode == 'poisson':
        await asyncio.sleep(random.expovariate(rps))
      else:
        raise ValueError("Invalid mode. Choose 'ideal' or 'poisson'.")

    responses = await asyncio.gather(*tasks)
    return responses, sum(r.latency for r in responses if r.status_code == 200) / rps if rps else 0.0


def compute_metrics(results: List[RequestResult], wall_time: float) -> dict:
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


async def main():
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


if __name__ == '__main__':
  asyncio.run(main())
