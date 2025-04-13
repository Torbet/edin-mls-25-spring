import time
from collections import deque
import threading

class RequestQueue:
    def __init__(self):
        self.queue = deque()  # Queue to store tuples of (timestamp, request)
        self._lock = threading.Lock()  # Thread-safe lock for queue operations

    def add_request(self, request):
        # Store the request with its timestamp in a tuple
        timestamp = time.time()
        with self._lock:
            self.queue.append((timestamp, request))  # Add timestamp and request as a tuple

    def get_batch(self, max_size, max_wait_time):
        batch = []

        with self._lock:
            # Check if we have enough items for a batch
            if len(self.queue) >= max_size:
                for _ in range(max_size):
                    timestamp, request = self.queue.popleft()
                    batch.append(request)  # Efficient pop from front
                return batch

            now = time.time()
            
            # If not enough, check if the oldest request timed out
            if self._oldest_request_timed_out(now, max_wait_time):
                while self.queue and len(batch) < max_size:
                    timestamp, request = self.queue.popleft()
                    batch.append(request)  # Efficient pop from front
                return batch

        return batch

    def _oldest_request_timed_out(self, now, max_wait_time):
        if self.queue:
            timestamp, _ = self.queue[0]  # Get the timestamp of the oldest request
            if now - timestamp > max_wait_time:
                return True
        return False
