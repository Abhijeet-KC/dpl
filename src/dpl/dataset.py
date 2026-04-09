# get_dataloaders(root) function, batch_size and shuffle && return test, train and val_dataloader
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import random_split

transforms = v2.Compose([v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.1307, ), (0.3081, )),
                        v2.RandomRotation(degrees=20)])

def get_dataloaders(root, batch_size, shuffle, transforms):
    
    train_set = MNIST(root, train=True, download=True, transform=transforms)
    test_set = MNIST(root, train=False, download=True, transform=transforms)

    generator1 = torch.Generator().manual_seed(66)
    val_set, test_set = random_split(test_set, lengths=[0.5, 0.5], generator=generator1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_sampler=16, shuffle=False)
    
    return train_loader, val_loader, test_loader   