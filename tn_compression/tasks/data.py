"""Small offline fixtures and explicit public-dataset split/mask contracts."""

import hashlib

import torch
from torch.utils.data import Dataset


def split_indices(identifiers, *, seed=42):
    """Stable disjoint train/calibration/validation roles; test is separate upstream."""
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Dataset identifiers must be unique.")
    ordered = sorted(range(len(identifiers)), key=lambda index: hashlib.sha256(
        f"{seed}:{identifiers[index]}".encode()).hexdigest())
    first, second = int(len(ordered) * 0.7), int(len(ordered) * 0.8)
    return {"train": ordered[:first], "calibration": ordered[first:second], "validation": ordered[second:]}


def map_pet_mask(mask, *, binary=False):
    """Pet=1, background=2, border=3; binary excludes ambiguous border pixels."""
    if not ((mask >= 1) & (mask <= 3)).all():
        raise ValueError("Oxford Pet trimaps must contain labels 1, 2, 3.")
    if binary:
        result = (mask == 1).long()
        result[mask == 3] = 255
        return result
    return mask.long() - 1


class SyntheticVisionDataset(Dataset):
    """Learnable offline fixtures, never a source of real-world accuracy claims."""

    def __init__(self, task="segmentation", count=16, size=32, num_classes=3, seed=0, binary=False):
        self.task, self.count, self.size = task, count, size
        self.num_classes, self.seed, self.binary = num_classes, seed, binary

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(3, self.size, self.size, generator=generator)
        if self.task == "classification":
            label = index % self.num_classes
            image[label % 3] = 0.7 + image[label % 3] * 0.3
            return image, torch.tensor(label)
        classes = 2 if self.binary else self.num_classes
        mask = (image[0] * classes).long().clamp_max(classes - 1)
        return image, mask


class OxfordPetDataset(Dataset):
    def __init__(self, root, role="train", task="segmentation", size=256,
                 binary=False, download=False, seed=42, normalize=True):
        from torchvision.datasets import OxfordIIITPet
        self.dataset = OxfordIIITPet(root, split="test" if role == "test" else "trainval",
                                    target_types=["segmentation", "category"], download=download)
        self.task, self.size, self.binary, self.normalize = task, size, binary, normalize
        identifiers = [path.stem for path in self.dataset._images]
        self.indices = list(range(len(identifiers))) if role == "test" else split_indices(identifiers, seed=seed)[role]
        self.identifiers = [identifiers[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        import numpy as np
        from torchvision.transforms import functional as transforms
        from torchvision.transforms import InterpolationMode
        image, (mask, category) = self.dataset[self.indices[index]]
        image = transforms.to_tensor(transforms.resize(image, [self.size, self.size], antialias=True))
        if self.normalize:
            image = transforms.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.task == "classification":
            return image, torch.tensor(category)
        mask = transforms.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        return image, map_pet_mask(torch.from_numpy(np.array(mask, copy=True)), binary=self.binary)


class CocoDetectionDataset(Dataset):
    """Disjoint fixed COCO subsets; preserve original image coordinates and IDs."""

    def __init__(self, root, annotations, role="validation", limit=None, seed=42):
        from torchvision.datasets import CocoDetection
        self.dataset = CocoDetection(root, annotations)
        self.coco = self.dataset.coco
        roles = {"calibration": 0, "validation": 1, "test": 2}
        if role not in roles:
            raise ValueError("Detection supports calibration, validation and test subsets.")
        ordered = sorted(range(len(self.dataset)), key=lambda index: hashlib.sha256(
            f"{seed}:{self.dataset.ids[index]}".encode()).hexdigest())
        self.indices = ordered[roles[role]::3][:limit]
        self.identifiers = [str(self.dataset.ids[index]) for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        from torchvision.transforms.functional import to_tensor
        original_index = self.indices[index]
        image, annotations = self.dataset[original_index]
        boxes, labels = [], []
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            if width > 0 and height > 0:
                boxes.append([x, y, x + width, y + height])
                labels.append(annotation["category_id"])
        return to_tensor(image), {"image_id": self.dataset.ids[original_index],
                                 "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                 "labels": torch.tensor(labels, dtype=torch.long)}
