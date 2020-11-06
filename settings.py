import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class CIFARSetting:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

        self.train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.loss = nn.CrossEntropyLoss()
        self.train_dset = datasets.CIFAR10(root='./data/cifar', train=True,
                                                    download=True, transform=self.train_transforms)

        self.test_dset = datasets.CIFAR10(root='./data/cifar', train=False,
                                                download=True, transform=self.test_transforms)

class INET32Setting:
    def __init__(self):
        self.train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.train_dset = datasets.ImageNet(root='./data/inet', train=True,
                                                    download=False, transform=self.train_transforms)

        self.test_dset = datasets.ImageNet(root='./data/inet', train=False,
                                                download=False, transform=self.test_transforms)

class MNISTSetting:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()
        self.train_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dset = datasets.MNIST('../data', train=True, download=True,
                       transform=self.train_transforms)
        self.test_dset = datasets.MNIST('../data', train=False,
                       transform=self.test_transforms)
valid_exps = {'cifar':CIFARSetting(),'mnist':MNISTSetting()}            