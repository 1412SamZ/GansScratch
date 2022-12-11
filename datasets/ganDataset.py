# pytorch lightning for better use 
import pytorch_lightning as pl 

# import core packages
import os
import torch
import torchvision 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader, random_split 
import torchvision.transforms as transforms 

# import the classical dataset MNIST!
from torchvision.datasets import MNIST  



# define the random seed for this experiment so you can always get the same selected data
random_seed = 42
torch.manual_seed(random_seed) 

# define experiment settings
BATCH_SIZE=64
NUM_WORKERS=int(os.cpu_count()/2) # use half cpu cores

DATA_DIR="../data"



class MNISTDataModule (pl.LightningDataModule):
    def __init__(self, data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        
    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None) -> None:
        if stage == 'fit' or not stage:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # split for train and valid datasets
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
        if stage == "test" or not stage:
            # init test dataset
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            
    # define functions to return train_valid_test dataset
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)