from dpl.stem import MNISTStem
from dpl.data.base import DataModule
import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split
  

class MNISTDataModule(DataModule):
    
    def __init__(
        self,
        root, 
        train = True,
        download = True,
        batch_size = 32,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
        use_stem = True 
    ):
        
        super().__init__()
        
        self.root = root
        self.train = train
        self.download = download
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.stem = MNISTStem() if use_stem else None
        
    def prepare_data(self):
        
        self.train_dataset = MNIST(root=self.root, train=self.train, download=self.download, transform=self.stem)
    
        # make self.train_dataset and self.test_dataset
        
    def setup(self):
        
        # split self.test_dataset into self.val_dataset and self.test_dataset
        
        self.test_dataset = MNIST(root=self.root, train=self.train, download=self.download, transform=self.stem)

        generator1 = torch.Generator().manual_seed(66)
        
        self.val_dataset, self.test_dataset = random_split(dataset=self.test_dataset, lengths=[0.5, 0.5], generator=generator1)