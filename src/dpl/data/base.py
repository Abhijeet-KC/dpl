from torch.utils.data import DataLoader

class DataModule():
    
    def __init__(
        self,
        root, 
        batch_size = 32,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
    ):
        
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def prepare_data(self):
        
        raise NotImplementedError("Implement Me!")
    
    def setup(self):
        
        raise NotImplementedError("Implement Me!")
    
    def train_dataloader(self):
        
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
        )

    def val_dataloader(self):
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        
    def test_dataloader(self):
        
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )