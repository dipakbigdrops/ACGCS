import pytest
from api.rate_limit import SlidingWindowRateLimiter


def test_allowed_when_under_limit():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_sec=60)
    allowed, _ = limiter.is_allowed("key1")
    assert allowed is True
    allowed, _ = limiter.is_allowed("key1")
    assert allowed is True


def test_rejected_when_over_limit():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_sec=60)
    limiter.is_allowed("key2")
    limiter.is_allowed("key2")
    allowed, retry_after = limiter.is_allowed("key2")
    assert allowed is False
    assert retry_after >= 0


def test_empty_key_always_allowed():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_sec=60)
    allowed, _ = limiter.is_allowed("")
    assert allowed is True
