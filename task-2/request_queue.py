import time
from collections import deque
import threading


class RequestQueue:
    def __init__(self):
        self.queue = deque()  # Queue to store tuples of (timestamp, request)
        self._lock = threading.Lock()  # Thread-safe lock for queue operations

    def add_request(self, request):
        """
        Adds a request to the queue with the current timestamp.
        """
        timestamp = time.time()  # Capture the current time
        with self._lock:
            self.queue.append((timestamp, request))  # Add timestamp and request as a tuple

    def get_batch(self, max_size, max_wait_time):
        """
        Retrieves a batch of requests from the queue with a maximum size. If there are not enough requests,
        it waits for the oldest requests to time out based on `max_wait_time`.
        """
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
                # Process requests until we have a batch of size max_size or empty the queue
                while self.queue and len(batch) < max_size:
                    timestamp, request = self.queue.popleft()
                    batch.append(request)  # Efficient pop from front
                return batch

        return batch  # If no batch, return an empty list

    def _oldest_request_timed_out(self, now, max_wait_time):
        """
        Helper method to check if the oldest request in the queue has timed out.
        """
        if self.queue:
            timestamp, _ = self.queue[0]  # Get the timestamp of the oldest request
            if now - timestamp > max_wait_time:  # Check if the oldest request timed out
                return True
        return False
