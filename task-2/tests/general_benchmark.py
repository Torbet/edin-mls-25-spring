import asyncio
import httpx
import time
import random
import argparse
import traceback
import csv
from typing import List, Tuple
from statistics import mean, quantiles
import os


# Constants
TIMEOUT = 60.0
QUERY_LIST = [
    "Which animals are commonly kept as pets?",
    "Which animals are known for flying or hovering?",
    "What animal is usually domesticated?"
]
K_VALUE = 2
SEED_VALUE = 123

ENDPOINTS_MAP = {
    "default": "http://localhost:8000/rag",
    "lb_server": "http://localhost:9000/assign"
}

class RequestMetrics:
    def __init__(self, response_time: float, code: int):
        self.response_time = response_time
        self.code = code


async def make_request(client: httpx.AsyncClient, endpoint: str, server_type: str) -> RequestMetrics:
    query = random.choice(QUERY_LIST)  # Pick a random query from the list
    request_payload = {"query": query, "k": K_VALUE}
    start_time = time.time()
    
    try:
        if server_type == "lb_server":
            lb_response = await client.get(endpoint)
            backend_server = lb_response.json().get("backend")
            if not backend_server:
                raise ValueError("Backend server not assigned.")
            response = await client.post(backend_server, json=request_payload)
        else:
            response = await client.post(endpoint, json=request_payload)

        response_time = time.time() - start_time
        return RequestMetrics(response_time, response.status_code)
    
    except Exception as err:
        print(f"[ERROR] Request error: {err}")
        traceback.print_exc()
        return RequestMetrics(response_time=float('inf'), code=0)


async def run_load_test(
    requests_per_second: int,
    total_requests: int,
    timing_mode: str,
    target_endpoint: str,
    server_type: str
) -> Tuple[List[RequestMetrics], float]:
    if timing_mode == "poisson":
        random.seed(SEED_VALUE)

    metrics = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = []

        for _ in range(total_requests):
            task = asyncio.create_task(make_request(client, target_endpoint, server_type))
            tasks.append(task)

            # Throttle request rate based on mode
            if timing_mode == "ideal":
                await asyncio.sleep(1 / requests_per_second)
            elif timing_mode == "poisson":
                await asyncio.sleep(random.expovariate(requests_per_second))
            else:
                raise ValueError("Invalid timing mode. Use 'ideal' or 'poisson'.")

        responses = await asyncio.gather(*tasks)
        return responses, sum(r.response_time for r in responses if r.code == 200) / requests_per_second if requests_per_second else 0.0


def calculate_metrics(results: List[RequestMetrics], total_time: float) -> dict:
    successful_times = [r.response_time for r in results if r.code == 200]
    failed_count = len([r for r in results if r.code != 200])

    return {
        "total_requests": len(results),
        "failed_requests": failed_count,
        "average_latency": mean(successful_times) if successful_times else None,
        "p95_latency": quantiles(successful_times, n=20)[18] if len(successful_times) >= 20 else None,
        "throughput_rps": len(successful_times) / total_time if total_time > 0 else None
    }


def save_results_to_csv(metrics: dict, filename: str):
    # Define the path to the test-results folder
    folder_path = "test-results"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Define the header for the CSV file
    header = ["Metric", "Value"]

    # Data to be written to the CSV file
    data = list(metrics.items())

    # Open the file in write mode and write the header and data
    with open(file_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        writer.writerows(data)  # Write the metric data

    print(f"Results saved to {file_path}")


    print(f"Results saved to {filename}")

def save_combined_results(metrics_list: List[dict], filename: str):
    folder_path = "test-results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, filename)

    # Use all keys from the first metrics dict as headers
    headers = ["rps"] + [key for key in metrics_list[0] if key != "rps"]

    with open(file_path, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metrics_list)

    print(f"\n✅ Combined results saved to {file_path}")


async def execute_test():
    parser = argparse.ArgumentParser(description="Load Test Automation")
    parser.add_argument("--timing_mode", choices=["ideal", "poisson"], default="ideal", help="Request timing method")
    parser.add_argument("--target", choices=list(ENDPOINTS_MAP.keys()), default="default", help="API endpoint to test")
    parser.add_argument("--total_requests", type=int, required=True, help="Total number of requests to send")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file name")

    args = parser.parse_args()
    target_url = ENDPOINTS_MAP[args.target]

    rps_list = [5,10,15,20,25,30,35,40,45,50,75,100]  # RPS values to test
    all_metrics = []

    for rps in rps_list:
        print(f"\n=== Running test at {rps} RPS ===")
        print(f"Test Parameters:\n - Timing Mode: {args.timing_mode}\n - Target: {args.target}\n - Total Requests: {args.total_requests}")

        start_time = time.time()
        results, simulated_duration = await run_load_test(
            rps, args.total_requests, args.timing_mode, target_url, args.target
        )
        total_time = time.time() - start_time

        metrics = calculate_metrics(results, total_time)
        metrics["rps"] = rps  # Add RPS as a metric field

        all_metrics.append(metrics)

        print("Cooldown before next test...\n")
        await asyncio.sleep(10)

    # Save combined results
    save_combined_results(all_metrics, args.output_file)



if __name__ == "__main__":
    asyncio.run(execute_test())
