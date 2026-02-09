import logging
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_request_latency: Dict[str, List[float]] = defaultdict(list)
_request_errors: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
_request_count: Dict[str, int] = defaultdict(int)
_ocr_confidence_samples: List[float] = []
_semantic_confidence_samples: List[float] = []
_max_samples = 10000
_lock = threading.Lock()


def record_latency(endpoint: str, duration_sec: float):
    if not endpoint:
        return
    with _lock:
        _request_count[endpoint] += 1
        lst = _request_latency[endpoint]
        lst.append(duration_sec)
        if len(lst) > _max_samples:
            lst.pop(0)


def record_error(endpoint: str, status_code: int):
    with _lock:
        _request_errors[endpoint][status_code] += 1


def record_ocr_confidence(confidence: float):
    with _lock:
        _ocr_confidence_samples.append(confidence)
        if len(_ocr_confidence_samples) > _max_samples:
            _ocr_confidence_samples.pop(0)


def record_semantic_confidence(confidence: float):
    with _lock:
        _semantic_confidence_samples.append(confidence)
        if len(_semantic_confidence_samples) > _max_samples:
            _semantic_confidence_samples.pop(0)


def _percentile(sorted_arr: List[float], p: float) -> Optional[float]:
    if not sorted_arr:
        return None
    k = (len(sorted_arr) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_arr) else f
    return sorted_arr[f] + (k - f) * (sorted_arr[c] - sorted_arr[f]) if c > f else sorted_arr[f]


def get_metrics() -> dict:
    with _lock:
        out = {
            "request_count": dict(_request_count),
            "request_errors": {k: dict(v) for k, v in _request_errors.items()},
            "latency_seconds": {},
            "ocr_confidence": {},
            "semantic_confidence": {},
        }
        for endpoint, lst in _request_latency.items():
            if not lst:
                continue
            s = sorted(lst)
            out["latency_seconds"][endpoint] = {
                "p50": _percentile(s, 50),
                "p95": _percentile(s, 95),
                "p99": _percentile(s, 99),
                "count": len(lst),
            }
        if _ocr_confidence_samples:
            s = sorted(_ocr_confidence_samples)
            out["ocr_confidence"] = {"p50": _percentile(s, 50), "p95": _percentile(s, 95), "count": len(s)}
        if _semantic_confidence_samples:
            s = sorted(_semantic_confidence_samples)
            out["semantic_confidence"] = {"p50": _percentile(s, 50), "p95": _percentile(s, 95), "count": len(s)}
    return out


def prometheus_export() -> str:
    lines = []
    with _lock:
        for endpoint, count in _request_count.items():
            name = endpoint.replace("/", "_").strip("_") or "root"
            lines.append(f'http_requests_total{{endpoint="{endpoint}"}} {count}')
        for endpoint, errs in _request_errors.items():
            for code, n in errs.items():
                lines.append(f'http_requests_errors_total{{endpoint="{endpoint}",status="{code}"}} {n}')
        for endpoint, lst in _request_latency.items():
            if not lst:
                continue
            s = sorted(lst)
            p50 = _percentile(s, 50)
            p95 = _percentile(s, 95)
            if p50 is not None:
                lines.append(f'http_request_duration_seconds{{endpoint="{endpoint}",quantile="0.5"}} {p50}')
            if p95 is not None:
                lines.append(f'http_request_duration_seconds{{endpoint="{endpoint}",quantile="0.95"}} {p95}')
    return "\n".join(lines) + "\n"
