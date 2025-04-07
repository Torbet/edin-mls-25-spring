import argparse
import time
import requests
import threading
import json

# Default base URL of your FastAPI RAG service
BASE_URL = 'http://0.0.0.0:8000/rag'


# Function to send a request to the RAG service
def send_request(query, rate, duration):
  start_time = time.time()
  requests_sent = 0

  while time.time() - start_time < duration:
    payload = {'query': query}
    try:
      response = requests.post(BASE_URL, json=payload)
      if response.status_code == 200:
        print(f'Success: {response.json()}')
      else:
        print(f'Failed to get response: {response.status_code}')
    except Exception as e:
      print(f'Error during request: {e}')
    requests_sent += 1
    time.sleep(1 / rate)  # Control the request rate

  print(f'Sent {requests_sent} requests in {duration} seconds at a rate of {rate} rps')


# Function to handle argument parsing
def get_args():
  parser = argparse.ArgumentParser(description='Test FastAPI RAG service with varying request rates')
  parser.add_argument('--query', type=str, required=True, help='Query to send to the RAG service')
  parser.add_argument('--rate', type=int, default=1, help='Requests per second')
  parser.add_argument('--duration', type=int, default=10, help='Duration for the test in seconds')
  parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads to simulate load')

  return parser.parse_args()


# Main function to run the test
def main():
  args = get_args()

  threads = []
  for i in range(args.threads):
    # Create a new thread for sending requests
    thread = threading.Thread(target=send_request, args=(args.query, args.rate, args.duration))
    threads.append(thread)
    thread.start()

  # Wait for all threads to finish
  for thread in threads:
    thread.join()

  print('All requests have been processed.')


if __name__ == '__main__':
  main()


# Server and script running on GPU cluster

# python serving_rag &                 - (Include '&' to run it in the background of the same terminal)
# python testing_script.py --query "Which animals can hover in the air?" --rate 10 --duration 60 --threads 1  (will attempt to send 10 requests per second per thread - current issue is it awaits the response so request rate per thread is not optimal)
# ps aux | grep serving_rag.py            - (Allows you to see PID so u can kill the process when your done)
# kill <pid>
