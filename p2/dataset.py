import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


# image dataset code is modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/datasets.py
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        if mode == "train":
            self.train_input = []
            self.train_mask = []

            for f in os.listdir(root):
                number = int(f.split('.')[0].split('_')[0])
                input_or_mask = f.split('.')[0].split('_')[1]
                if input_or_mask == "sat":
                    self.train_input.append(os.path.join(root, f))
                elif input_or_mask == "mask":
                    self.train_mask.append(os.path.join(root, f))
                        
            self.train_input = sorted(self.train_input)
            self.train_mask = sorted(self.train_mask)


        elif mode == "val":
            self.val_input= []
            self.val_mask = [] 

            for f in os.listdir(root):
                number = int(f.split('.')[0].split('_')[0])
                input_or_mask = f.split('.')[0].split('_')[1]
                if input_or_mask == "sat":
                    self.val_input.append(os.path.join(root, f))
                elif input_or_mask == "mask":
                    self.val_mask.append(os.path.join(root, f))
            
            self.val_input = sorted(self.val_input)
            self.val_mask = sorted(self.val_mask)
            
        elif mode == "test":
            self.test_input = []
            self.test_input = glob.glob(os.path.join(root, "*"))
            self.test_input = sorted(self.test_input)

    def encode_label(self, mask_img):
        self.totensor = transforms.Compose([transforms.ToTensor()])
        mask_img = self.totensor(mask_img) 
        mask_img = mask_img.permute(1, 2, 0)
        mask_img = 4 * mask_img[:, :, 0] + 2 * mask_img[:, :, 1] + 1 * mask_img[:, :, 2]
        mask = torch.zeros(mask_img.size(), dtype=torch.long)

        mask[mask_img == 3] = 0  # (Cyan: 011) Urban land 
        mask[mask_img == 6] = 1  # (Yellow: 110) Agriculture land 
        mask[mask_img == 5] = 2  # (Purple: 101) Rangeland 
        mask[mask_img == 2] = 3  # (Green: 010) Forest land 
        mask[mask_img == 1] = 4  # (Blue: 001) Water 
        mask[mask_img == 7] = 5  # (White: 111) Barren land 
        mask[mask_img == 0] = 6  # (Black: 000) Unknown 

        return mask


    def __getitem__(self, idx):
        if self.mode == "train":
            input_img = Image.open(self.train_input[idx])
            mask_img = Image.open(self.train_mask[idx]) 
            input_img = self.transform(input_img)
            ## rgb -> label
            mask_img = self.encode_label(mask_img)
            return input_img, mask_img 
        elif self.mode == "val":
            input_img = Image.open(self.val_input[idx])
            mask_img = Image.open(self.val_mask[idx]) 
            input_img = self.transform(input_img)
            ## rgb -> label
            mask_img = self.encode_label(mask_img)
            return input_img, mask_img 

        elif self.mode == "test":
            input_img = Image.open(self.test_input[idx])
            input_img = self.transform(input_img)
            return input_img, self.test_input[idx]

    def __len__(self):
        if self.mode == "train":
            return len(self.train_input)
        elif self.mode == "val":
            return len(self.val_input)
        elif self.mode == "test":
            return len(self.test_input)


#transforms_ = [
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#]
#totensor = transforms.Compose(transforms_)


#train_imagedataset = ImageDataset("../hw1_data/p2_data/validation_input", transforms_= transforms_ , mode="test")
#train_dataloader = DataLoader(train_imagedataset, batch_size = 1)
#print(len(train_dataloader))
#for data in train_dataloader:
#    print(data.size())

