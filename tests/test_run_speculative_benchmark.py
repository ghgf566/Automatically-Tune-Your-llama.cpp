"""Comparison guards for the controlled server benchmark."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_speculative_benchmark import _effective_gpu_layers_match


class EffectiveGpuLayersTests(unittest.TestCase):
    def test_missing_layer_observation_is_unknown(self):
        self.assertIsNone(_effective_gpu_layers_match(None, None))
        self.assertIsNone(_effective_gpu_layers_match(16, None))

    def test_observed_layer_counts_must_match(self):
        self.assertTrue(_effective_gpu_layers_match(16, 16))
        self.assertFalse(_effective_gpu_layers_match(16, 15))


if __name__ == "__main__":
    unittest.main()
