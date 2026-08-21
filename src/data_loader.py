import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch


def _strong_train_tf(img_size):
    # Stronger but safe augmentations for otoscopic images
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _light_train_tf(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _eval_tf(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(data_dir, batch_size=32, img_size=224, use_strong_aug=False):
    """
    Returns dataloaders for train/val/test + sizes + class names.
    Adds a WeightedRandomSampler on train to mitigate class imbalance.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_tf = (
        _strong_train_tf(img_size) if use_strong_aug else _light_train_tf(img_size)
    )
    eval_tf = _eval_tf(img_size)

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=train_tf),
        "val": datasets.ImageFolder(val_dir, transform=eval_tf),
        "test": datasets.ImageFolder(test_dir, transform=eval_tf),
    }

    # Weighted sampler for train (handles class imbalance)
    targets = image_datasets["train"].targets
    targets = torch.tensor(targets)
    class_count = torch.bincount(targets).float()
    class_weight = class_count.sum() / (class_count + 1e-6)
    sample_weight = class_weight[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weight, num_samples=len(sample_weight), replacement=True
    )

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
        ),
        "val": DataLoader(
            image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4
        ),
        "test": DataLoader(
            image_datasets["test"], batch_size=batch_size, shuffle=False, num_workers=4
        ),
    }

    dataset_sizes = {k: len(v.dataset) for k, v in dataloaders.items()}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names
