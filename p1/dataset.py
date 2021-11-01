import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# image dataset code is modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/datasets.py

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode

        if mode == "train":
            self.train_files = []
            for f in os.listdir(root):
                self.train_files.append(os.path.join(root, f))
            self.train_files = sorted(self.train_files)
        elif mode == "val":
            self.val_files = []
            for f in os.listdir(root):
                self.val_files.append(os.path.join(root, f))
            self.val_files = sorted(self.val_files)
        elif mode == "test":
            self.test_files = []
            for f in os.listdir(root):
                self.test_files.append(os.path.join(root, f))
            self.test_files = sorted(self.test_files)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = Image.open(self.train_files[idx])
            img = self.transform(img)
            label = int(self.train_files[idx].split('/')[-1].split('.')[0].split('_')[0])
            return img, label
        
        elif self.mode == "val":
            img = Image.open(self.val_files[idx])
            img = self.transform(img)
            label = int(self.val_files[idx].split('/')[-1].split('.')[0].split('_')[0])
            return img, label

        elif self.mode == "test":
            img = Image.open(self.test_files[idx])
            img = self.transform(img)
            return img, self.test_files[idx]

    def __len__(self):
        if self.mode == "train":
            return len(self.train_files)
        
        elif self.mode == "val":
            return len(self.val_files)

        elif self.mode == "test":
            return len(self.test_files)

#transforms_ = [
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#]
#
#test_dataset = ImageDataset("../hw1_data/p1_data/val_50", transforms_, mode="val")
#test_dataloader = DataLoader(test_dataset, batch_size=2)
#for data in test_dataloader:
#    import pdb; pdb.set_trace()
