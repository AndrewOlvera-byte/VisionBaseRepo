from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10DataModule:
    """Minimal dataset wrapper that offers train/val(/test) dataloaders."""

    def __init__(self, root: str = "./data", batch_size: int = 64, num_workers: int = 4):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        _mean = (0.4914, 0.4822, 0.4465)
        _std = (0.2023, 0.1994, 0.2010)

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_mean, _std),
            ]
        )
        self.transform_eval = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(_mean, _std)]
        )

        # These will be initialised lazily in setup()
        self._train_set = None
        self._val_set = None
        self._test_set = None

    def _setup(self):
        if self._train_set is None:
            full_train = datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=self.transform_train
            )
            self._train_set, self._val_set = torch.utils.data.random_split(
                full_train, [45000, 5000]
            )
            # Apply eval transform to val subset
            self._val_set.dataset.transform = self.transform_eval

        if self._test_set is None:
            self._test_set = datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=self.transform_eval
            )

    # Public API
    def train_dataloader(self) -> DataLoader:
        self._setup()
        return DataLoader(
            self._train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        self._setup()
        return DataLoader(
            self._val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        self._setup()
        return DataLoader(
            self._test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        ) 