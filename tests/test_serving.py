import json

import pytest

from tn_compression.serving import read_events


def event(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n"


def test_stream_counts_actual_tokens_not_content_events():
    lines = [event({"choices": [{"text": "two tokens"}]}),
             event({"choices": [{"text": " more"}]}),
             event({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}), b"data: [DONE]\n"]
    clock = iter([11.0, 13.0, 13.5])
    result = read_events(lines, started=10.0, clock=lambda: next(clock))
    assert result["ttft_seconds"] == 1.0 and result["tpot_seconds"] == 1.0
    assert result["latency_seconds"] == 3.5


def test_stream_failure_is_not_a_successful_timing():
    with pytest.raises(ValueError, match="Incomplete stream"):
        read_events([event({"choices": [{"text": "partial"}]})], started=0, clock=lambda: 1)
