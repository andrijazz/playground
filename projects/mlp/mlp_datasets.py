import os
import torch
from torchvision import transforms, datasets


def create_train_and_val_datasets(dataset):
    if dataset == 'MNIST':
        dataset = datasets.MNIST(os.getenv('DATASETS'), train=True, download=True, transform=transforms.ToTensor())
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[len(dataset) - 100, 100])
        return train_dataset, val_dataset
    else:
        exit('Unsupported dataset {}'.format(dataset))


def create_test_dataset(dataset):
    if dataset == 'MNIST':
        test_dataset = datasets.MNIST(os.getenv('DATASETS'), train=False, download=True,
                                      transform=transforms.ToTensor())
        return test_dataset
    else:
        exit('Unsupported dataset {}'.format(dataset))

