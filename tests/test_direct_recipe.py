"""Portable direct-recipe artifacts must reload and remain honest about scope."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from tn_compression.checkpoints import construct_module, describe_module, load_bundle, save_bundle
from tn_compression.direct_recipe import apply_direct_recipe, validate_recipe
from tn_compression.packed_quantization import PackedInt8Conv2d, PackedInt8Linear


class DirectRecipeTests(unittest.TestCase):
    def test_int8_storage_roundtrip_for_linear_and_conv(self):
        for original, packed_type, sample in (
            (nn.Linear(32, 16), PackedInt8Linear, torch.randn(2, 32)),
            (nn.Conv2d(3, 16, 3, padding=1), PackedInt8Conv2d, torch.randn(2, 3, 8, 8)),
        ):
            with self.subTest(type=type(original).__name__):
                packed = packed_type.from_linear(original) if isinstance(original, nn.Linear) else packed_type.from_conv(original)
                rebuilt = construct_module(describe_module(packed))
                rebuilt.load_state_dict(packed.state_dict(), strict=True)
                self.assertTrue(torch.allclose(packed(sample), rebuilt(sample)))
                before = sum(t.numel() * t.element_size() for t in original.state_dict().values())
                after = sum(t.numel() * t.element_size() for t in packed.state_dict().values())
                self.assertLess(after, before)

    def test_mixed_vision_recipe_bundle_reloads(self):
        from tn_compression.models import load_model

        spec = {"source": "torchvision", "name": "resnet18", "task": "classification",
                "kwargs": {"num_classes": 3}, "weights": None}
        torch.manual_seed(9)
        model = load_model(spec)
        recipe = {"methods": ["tensor_decomposition", "quantization"], "svd_energy": 0.8,
                  "quantization_bits": 8, "include": ["layer1.0.conv2*"]}
        report = apply_direct_recipe(model, recipe, task="classification")
        self.assertEqual(report["applied_methods"], ["tensor_decomposition", "quantization"])
        self.assertLess(report["compressed_tensor_bytes"], report["original_tensor_bytes"])
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            save_bundle(model, bundle)
            loaded, _ = load_bundle(bundle)
            sample = torch.randn(1, 3, 32, 32)
            model.eval(), loaded.eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(sample), loaded(sample), atol=1e-5, rtol=1e-5))

    def test_unsupported_vision_pruning_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Llama/Qwen"):
            validate_recipe({"methods": ["pruning"], "pruning_retention": 0.8}, "classification")


if __name__ == "__main__":
    unittest.main()
