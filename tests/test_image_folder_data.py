"""Real labeled image splits retain class mapping and example identities."""

import tempfile
import unittest
from pathlib import Path


class ImageFolderTests(unittest.TestCase):
    def test_explicit_splits_and_classes(self):
        from PIL import Image
        from tn_compression.tasks.data import LabeledImageFolder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "calibration", "validation", "test"):
                for name, color in (("a", "red"), ("b", "blue")):
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    Image.new("RGB", (8, 8), color).save(folder / "example.png")
            dataset = LabeledImageFolder(root, "test", size=16)
            self.assertEqual(dataset.classes, ["a", "b"])
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(dataset[0][0].shape), (3, 16, 16))
            self.assertTrue(all(item.startswith("test/") for item in dataset.identifiers))
            (root / "validation" / "b" / "example.png").unlink()
            # Empty categories are rejected by ImageFolder before evaluation.
            with self.assertRaises(Exception):
                LabeledImageFolder(root, "validation")


if __name__ == "__main__":
    unittest.main()
