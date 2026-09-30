"""A saved direct recipe survives worker processes and final verification."""

import tempfile
import unittest
from pathlib import Path

from tn_compression.ui.jobs import JobManager, read_json
from tn_compression.ui.server import PRESETS


class DirectUIEndToEndTests(unittest.TestCase):
    def test_synthetic_mixed_recipe_reloads_and_compares(self):
        config = read_json(PRESETS)[0]["config"]
        config["analysis"].pop("target_size_mb", None)
        config["recipe"] = {"methods": ["tensor_decomposition", "quantization"],
                            "svd_energy": 0.8, "quantization_bits": 8,
                            "include": ["layer1.0.conv1*"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = JobManager(root / "runs", root / "data")
            job = manager.submit({"config": config, "mode": "compress", "device": "cpu"})
            manager.pool.shutdown(wait=True)
            result = manager.get(job["id"])
            self.assertEqual(result["status"], "completed", f"{result.get('error')}\n{result.get('logs')}")
            self.assertEqual(result["results"]["comparison"]["status"], "synthetic_only")
            self.assertEqual(result["results"]["direct"]["applied_methods"],
                             ["tensor_decomposition", "quantization"])
            self.assertTrue((root / "runs" / job["id"] / "compressed" / "bundle" / "weights.pt").is_file())


if __name__ == "__main__":
    unittest.main()
