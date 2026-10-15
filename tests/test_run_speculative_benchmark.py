"""Comparison guards for the controlled server benchmark."""
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_speculative_benchmark import (
    BenchmarkError,
    _effective_gpu_layers_match,
    _request,
    _summarize_scenario,
    _summary,
)


class _Response:
    status = 200

    def __init__(self, lines):
        self.lines = iter(line.encode("utf-8") for line in lines)

    def readline(self, _limit=None):
        return next(self.lines, b"")


class _Connection:
    def __init__(self, response):
        self.response = response

    def request(self, *_args, **_kwargs):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        pass


class EffectiveGpuLayersTests(unittest.TestCase):
    def test_missing_layer_observation_is_unknown(self):
        self.assertIsNone(_effective_gpu_layers_match(None, None))
        self.assertIsNone(_effective_gpu_layers_match(16, None))

    def test_observed_layer_counts_must_match(self):
        self.assertTrue(_effective_gpu_layers_match(16, 16))
        self.assertFalse(_effective_gpu_layers_match(16, 15))
        self.assertFalse(_effective_gpu_layers_match(16, 16, 66, 67))
        self.assertIsNone(_effective_gpu_layers_match(16, 16, 66, None))

    def test_even_sample_median_uses_both_middle_values(self):
        self.assertEqual(_summary([1, 4])["median"], 2.5)

    def test_unknown_mtp_counters_remain_unknown(self):
        report = _summarize_scenario({"requests": [{
            "warmup": False, "drafted_tokens": None, "accepted_tokens": None,
            "latency_ms": 10, "ttft_ms": 2,
            "prompt_tokens_per_second": 3, "decode_tokens_per_second": 4,
        }]})
        self.assertIsNone(report["drafted_tokens"])
        self.assertIsNone(report["accepted_tokens"])
        self.assertIsNone(report["acceptance_rate"])

    def test_request_requires_complete_stream_and_valid_metrics(self):
        timings = {
            "prompt_n": 20, "prompt_ms": 100, "prompt_per_second": 200,
            "predicted_n": 8, "predicted_ms": 1000,
            "predicted_per_second": 8,
        }
        lines = [
            'data: {"content":"x"}\n',
            "data: {\"timings\":" + json.dumps(timings) + "}\n",
            "data: [DONE]\n",
        ]
        connection = _Connection(_Response(lines))
        with patch("run_speculative_benchmark.http.client.HTTPConnection",
                   return_value=connection):
            with self.assertRaisesRegex(BenchmarkError, "drafted/accepted"):
                _request(1234, n_predict=8, seed=1, mode="embedded_mtp")

        connection = _Connection(_Response(lines[:-1]))
        with patch("run_speculative_benchmark.http.client.HTTPConnection",
                   return_value=connection):
            with self.assertRaisesRegex(BenchmarkError, "terminal timing event"):
                _request(1234, n_predict=8, seed=1, mode="off")

    def test_request_accepts_terminal_timing_event_without_done_sentinel(self):
        timings = {
            "prompt_n": 20, "prompt_ms": 100, "prompt_per_second": 200,
            "predicted_n": 8, "predicted_ms": 1000,
            "predicted_per_second": 8,
        }
        connection = _Connection(_Response([
            'data: {"content":"x"}\n',
            "data: {\"stop\":true,\"timings\":" + json.dumps(timings) + "}\n",
        ]))
        with patch("run_speculative_benchmark.http.client.HTTPConnection",
                   return_value=connection):
            row = _request(1234, n_predict=8, seed=1, mode="off")
        self.assertEqual(row["predicted_tokens"], 8)

    def test_request_rejects_non_object_events(self):
        connection = _Connection(_Response([
            'data: [1,2]\n',
        ]))
        with patch("run_speculative_benchmark.http.client.HTTPConnection",
                   return_value=connection):
            with self.assertRaisesRegex(BenchmarkError, "not a JSON object"):
                _request(1234, n_predict=8, seed=1)

    def test_valid_mtp_request_preserves_counts(self):
        timings = {
            "prompt_n": 20, "prompt_ms": 100, "prompt_per_second": 200,
            "predicted_n": 8, "predicted_ms": 1000,
            "predicted_per_second": 8, "draft_n": 10,
            "draft_n_accepted": 4,
        }
        connection = _Connection(_Response([
            'data: {"content":"x"}\n',
            "data: {\"timings\":" + json.dumps(timings) + "}\n",
            "data: [DONE]\n",
        ]))
        with patch("run_speculative_benchmark.http.client.HTTPConnection",
                   return_value=connection):
            row = _request(1234, n_predict=8, seed=1, mode="embedded_mtp")
        self.assertEqual(row["drafted_tokens"], 10)
        self.assertEqual(row["accepted_tokens"], 4)

    def test_expired_request_budget_fails_before_connecting(self):
        with patch("run_speculative_benchmark.http.client.HTTPConnection") as connect:
            with self.assertRaisesRegex(BenchmarkError, "time budget expired"):
                _request(1234, n_predict=8, seed=1,
                         deadline=time.monotonic() - 1)
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
