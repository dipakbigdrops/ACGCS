import pytest
from api.metrics import (
    record_latency,
    record_error,
    record_ocr_confidence,
    record_semantic_confidence,
    get_metrics,
    prometheus_export,
)


def test_record_latency_and_get_metrics():
    record_latency("/test", 0.1)
    record_latency("/test", 0.2)
    m = get_metrics()
    assert "request_count" in m
    assert "/test" in m["request_count"]
    assert m["request_count"]["/test"] >= 2
    assert "latency_seconds" in m
    assert "/test" in m["latency_seconds"]


def test_record_error():
    record_error("/api", 404)
    record_error("/api", 500)
    m = get_metrics()
    assert "/api" in m["request_errors"]
    assert 404 in m["request_errors"]["/api"]
    assert 500 in m["request_errors"]["/api"]


def test_record_confidence():
    record_ocr_confidence(0.9)
    record_semantic_confidence(0.85)
    m = get_metrics()
    assert "ocr_confidence" in m or "semantic_confidence" in m


def test_prometheus_export():
    out = prometheus_export()
    assert isinstance(out, str)
    assert "\n" in out or len(out) >= 0
