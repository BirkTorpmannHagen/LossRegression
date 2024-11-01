import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.datasets import ImageFolder
import zipfile
from njord.utils.general import check_dataset, colorstr
from njord.utils.dataloaders import create_dataloader
from train import *


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148), #2048 with, 4096 without...
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            # transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])


        self.train_dataset = ImageFolder(
        self.data_dir,
        transform=train_transforms,
        )

        # Replace CelebA with your dataset
        self.val_dataset = ImageFolder(
        self.data_dir,
        transform=val_transforms,
)
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

# class VAEDataset(LightningDataModule):
#     """
#     PyTorch Lightning data module
#
#     Args:
#         data_dir: root directory of your dataset.
#         train_batch_size: the batch size to use during training.
#         val_batch_size: the batch size to use during validation.
#         patch_size: the size of the crop to take from the original images.
#         num_workers: the number of parallel workers to create to load data
#             items (see PyTorch's Dataloader documentation for more details).
#         pin_memory: whether prepared items should be loaded into pinned memory
#             or not. This can improve performance on GPUs.
#     """
#
#     def __init__(
#             self,
#             data_path: str,
#             train_batch_size: int = 16,
#             val_batch_size: int = 16,
#             patch_size: Union[int, Sequence[int]] = (256, 256),
#             num_workers: int = 0,
#             pin_memory: bool = False,
#             **kwargs,
#     ):
#         super().__init__()
#         self.opt =
#         self.data_dict = check_dataset(data_path)
#         self.train_data, self.val_data = self.data_dict['train'], self.data_dict['val']
#
#         self.batch_size = train_batch_size
#         self.train_batch_size = train_batch_size
#         self.val_batch_size = val_batch_size
#         self.patch_size = patch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#
#     def setup(self, stage: Optional[str] = None) -> None:
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                                # transforms.CenterCrop(148), #2048 with, 4096 without...
#                                                transforms.Resize(self.patch_size),
#                                                transforms.ToTensor(), ])
#
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                              # transforms.CenterCrop(148),
#                                              transforms.Resize(self.patch_size),
#                                              transforms.ToTensor(), ])
#
#         # todo interface this properly if continuing sampler work
#         self.train_loader, self.train_dataset = create_dataloader(
#             self.train_data,
#             self.patch_size,
#             self.train_batch_size // WORLD_SIZE,
#             32,
#             int(self.data_dict['nc']),
#             hyp=opt.hyp,
#             augment=True,
#             rank=LOCAL_RANK,
#             workers=self.num_workers,
#             prefix=colorstr('train: '),
#             shuffle=True)
#         for i, (imgs, targets, paths, _) in enumerate(self.train_loader):
#             print(i)
#             if i > 5:
#                 break
#
#         self.val_loader = create_dataloader(
#             self.val_data,
#             self.patch_size,
#             self.val_batch_size // WORLD_SIZE * 2,
#             32,
#             int(self.data_dict['nc']),
#             hyp=None,
#             rect=True,
#             rank=-1,
#             workers=self.num_workers * 2,
#             pad=0.5,
#             prefix=colorstr('val: '))[0]
