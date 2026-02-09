import logging
import time
import threading
from collections import deque
from typing import Deque, Dict, Tuple

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_sec: int):
        self._max_requests = max_requests
        self._window_sec = window_sec
        self._key_timestamps: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> None:
        q = self._key_timestamps[key]
        while q and q[0] < now - self._window_sec:
            q.popleft()
        if not q:
            self._key_timestamps.pop(key, None)

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        if not key:
            return True, 0
        now = time.monotonic()
        with self._lock:
            if key not in self._key_timestamps:
                self._key_timestamps[key] = deque()
            self._trim(key, now)
            if key not in self._key_timestamps:
                self._key_timestamps[key] = deque()
            q = self._key_timestamps[key]
            if len(q) >= self._max_requests:
                retry_after = int(self._window_sec)
                if q:
                    retry_after = max(0, int(q[0] + self._window_sec - now))
                return False, retry_after
            q.append(now)
            return True, 0
