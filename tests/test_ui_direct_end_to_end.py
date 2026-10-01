"""A saved direct recipe survives worker processes and final verification."""

import tempfile
import unittest
from pathlib import Path

from tn_compression.ui.jobs import JobManager, read_json
from tn_compression.ui.server import PRESETS


class DirectUIEndToEndTests(unittest.TestCase):
    def test_synthetic_segmentation_recipe_reloads_compares_and_exports(self):
        config = read_json(PRESETS)[0]["config"]
        config["task"] = "segmentation"
        config["model"] = {
            "source": "smp", "name": "Unet", "weights": None,
            "kwargs": {"encoder_name": "resnet18", "encoder_depth": 3,
                       "decoder_channels": [32, 16, 8], "classes": 3},
        }
        config["analysis"].pop("target_size_mb", None)
        config["recipe"] = {"methods": ["tensor_decomposition", "quantization"],
                            "svd_energy": 0.8, "quantization_bits": 8,
                            "include": ["encoder.layer1.0.conv1*"]}
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
            self.assertTrue((root / "runs" / job["id"] / "export" / "model.onnx").is_file())
            self.assertTrue(any(artifact["name"] == "export/model.onnx"
                                for artifact in result["artifacts"]))


if __name__ == "__main__":
    unittest.main()
