# Import necessary libraries for asynchronous HTTP requests, time tracking, and more
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


# Constants for the load test configuration
TIMEOUT = 60.0  # Timeout duration for each request
QUERY_LIST = [
    "Which animals are commonly kept as pets?",
    "Which animals are known for flying or hovering?",
    "What animal is usually domesticated?"
]  # List of queries to simulate

K_VALUE = 2  # A constant parameter for the query payload
SEED_VALUE = 123  # Seed value for random number generation (used in Poisson timing mode)

# Mapping of available API endpoints to test
ENDPOINTS_MAP = {
    "default": "http://localhost:8000/rag",  # Default API endpoint
    "lb_server": "http://localhost:9000/assign"  # Load balancer server endpoint
}

# Class to store the metrics of each request (response time and status code)
class RequestMetrics:
    def __init__(self, response_time: float, code: int):
        self.response_time = response_time  # Time taken for the request to complete
        self.code = code  # HTTP status code (e.g., 200 for success)

# Function to make the request asynchronously
async def make_request(client: httpx.AsyncClient, endpoint: str, server_type: str) -> RequestMetrics:
    query = random.choice(QUERY_LIST)  # Pick a random query from the list
    request_payload = {"query": query, "k": K_VALUE}  # Prepare request payload
    start_time = time.time()  # Record start time of the request
    
    try:
        if server_type == "lb_server":
            # If the server type is 'lb_server', send a request to the load balancer to get the backend server
            lb_response = await client.get(endpoint)
            backend_server = lb_response.json().get("backend")
            if not backend_server:
                raise ValueError("Backend server not assigned.")
            response = await client.post(backend_server, json=request_payload)  # Send the request to the backend server
        else:
            response = await client.post(endpoint, json=request_payload)  # Send request to the default endpoint

        response_time = time.time() - start_time  # Calculate the response time
        return RequestMetrics(response_time, response.status_code)  # Return the metrics

    except Exception as err:
        print(f"[ERROR] Request error: {err}")  # Handle any errors during the request
        traceback.print_exc()
        return RequestMetrics(response_time=float('inf'), code=0)  # Return a failed request with infinite response time

# Function to run the load test
async def run_load_test(
    requests_per_second: int,
    total_requests: int,
    timing_mode: str,
    target_endpoint: str,
    server_type: str
) -> Tuple[List[RequestMetrics], float]:
    if timing_mode == "poisson":
        random.seed(SEED_VALUE)  # Set the random seed for Poisson distribution if timing mode is Poisson

    metrics = []  # List to store the metrics of each request
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = []  # List to hold asynchronous task objects

        # Loop to create and run the specified number of requests
        for _ in range(total_requests):
            task = asyncio.create_task(make_request(client, target_endpoint, server_type))  # Create a request task
            tasks.append(task)

            # Throttle request rate based on timing mode
            if timing_mode == "ideal":
                await asyncio.sleep(1 / requests_per_second)  # Ideal rate: fixed interval between requests
            elif timing_mode == "poisson":
                await asyncio.sleep(random.expovariate(requests_per_second))  # Poisson rate: random interval between requests
            else:
                raise ValueError("Invalid timing mode. Use 'ideal' or 'poisson'.")

        responses = await asyncio.gather(*tasks)  # Execute all tasks concurrently
        # Return responses and the simulated duration (average time per request)
        return responses, sum(r.response_time for r in responses if r.code == 200) / requests_per_second if requests_per_second else 0.0


# Function to calculate various performance metrics from the test results
def calculate_metrics(results: List[RequestMetrics], total_time: float) -> dict:
    successful_times = [r.response_time for r in results if r.code == 200]  # Response times of successful requests
    failed_count = len([r for r in results if r.code != 200])  # Count of failed requests

    # Return a dictionary of calculated metrics
    return {
        "total_requests": len(results),
        "failed_requests": failed_count,
        "average_latency": mean(successful_times) if successful_times else None,
        "p95_latency": quantiles(successful_times, n=20)[18] if len(successful_times) >= 20 else None,  # 95th percentile latency
        "throughput_rps": len(successful_times) / total_time if total_time > 0 else None  # Requests per second throughput
    }

# Function to save the results of a single test to a CSV file
def save_results_to_csv(metrics: dict, filename: str):
    # Create a folder called 'test-results' if it doesn't exist
    folder_path = "test-results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the full path to the output CSV file
    file_path = os.path.join(folder_path, filename)
    
    # Header for the CSV file
    header = ["Metric", "Value"]

    # Write the metrics to the CSV file
    data = list(metrics.items())  # Convert the dictionary into a list of key-value pairs
    with open(file_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header row
        writer.writerows(data)  # Write the metric data

    print(f"Results saved to {file_path}")

# Function to save combined results from multiple tests to a CSV file
def save_combined_results(metrics_list: List[dict], filename: str):
    folder_path = "test-results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, filename)

    # Use all keys from the first metrics dict as headers
    headers = ["rps"] + [key for key in metrics_list[0] if key != "rps"]

    with open(file_path, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()  # Write the header row
        writer.writerows(metrics_list)  # Write the combined results

    print(f"\n Combined results saved to {file_path}")

# Main function to execute the load tests
async def execute_test():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load Test Automation")
    parser.add_argument("--timing_mode", choices=["ideal", "poisson"], default="ideal", help="Request timing method")
    parser.add_argument("--target", choices=list(ENDPOINTS_MAP.keys()), default="default", help="API endpoint to test")
    parser.add_argument("--total_requests", type=int, required=True, help="Total number of requests to send")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file name")

    args = parser.parse_args()  # Parse the arguments
    target_url = ENDPOINTS_MAP[args.target]  # Get the target endpoint URL based on the argument

    # Define a list of RPS values to test
    rps_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
    all_metrics = []  # List to hold the metrics from all tests

    # Loop over the RPS values to run tests at different request rates
    for rps in rps_list:
        print(f"\n=== Running test at {rps} RPS ===")
        print(f"Test Parameters:\n - Timing Mode: {args.timing_mode}\n - Target: {args.target}\n - Total Requests: {args.total_requests}")

        start_time = time.time()  # Record the start time of the test
        results, simulated_duration = await run_load_test(
            rps, args.total_requests, args.timing_mode, target_url, args.target
        )
        total_time = time.time() - start_time  # Calculate total test time

        # Calculate the performance metrics for the test
        metrics = calculate_metrics(results, total_time)
        metrics["rps"] = rps  # Add the RPS value to the metrics dictionary

        all_metrics.append(metrics)  # Append the metrics to the list

        print("Cooldown before next test...\n")
        await asyncio.sleep(10)  # Cooldown between tests

    # Save combined results to CSV
    save_combined_results(all_metrics, args.output_file)

# Entry point for the script
if __name__ == "__main__":
    asyncio.run(execute_test())  # Run the load test asynchronously
