import os
from typing import *

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from fastai.vision.all import *
from timm.data import create_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["imagenet", "cifar10", "mnist"]
DATASET_LOC = './data'


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenette":
        return 10
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10


def get_dataset(dataset: str, split: str) -> Dataset:
    if dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "imagenette":
        return _imagenette(split)


def _mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.MNIST(DATASET_LOC, train=False, download=True, transform=transforms.ToTensor())


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10(DATASET_LOC, train=False, download=True, transform=transforms.ToTensor())


def _imagenette(split: str) -> Dataset:
    path = untar_data(URLs.IMAGENETTE_160)
    if split == "train":
        return datasets.ImageFolder(root=(str(path) + '/train'), transform=transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.ImageFolder(root=(str(path) + '/val'), transform=transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ]))
