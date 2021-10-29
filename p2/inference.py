import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import Vgg16_bn, FCN_32, FCN_16, FCN_8, FCN_resnet50_8 
from dataset import ImageDataset 

import torch.nn as nn
import torch.nn.functional as F
import torch

def decode_label(label):
    #label -> rgb
    label_img = torch.zeros((512, 512, 3), dtype=torch.float)
    
    label_img[label == 0, :] = torch.Tensor([0, 1, 1])
    label_img[label == 1, :] = torch.Tensor([1, 1, 0])
    label_img[label == 2, :] = torch.Tensor([1, 0, 1])
    label_img[label == 3, :] = torch.Tensor([0, 1, 0])
    label_img[label == 4, :] = torch.Tensor([0, 0, 1])
    label_img[label == 5, :] = torch.Tensor([1, 1, 1])
    label_img[label == 6, :] = torch.Tensor([0, 0, 0])

    return label_img.permute(2, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./hw1_data/p2_data/validation_input", help="input directory")
    parser.add_argument("--output_dir", type=str, default="./hw1_data/p2_data/validation_output_FCN32", help="output directory")
    parser.add_argument("--model_path", type=str, default="./p2/checkpoint_FCN32/model_best.pth", help="model path")
    parser.add_argument("--model_name", type=str, default="FCN_32", help ="model name")
    opt = parser.parse_args()
    print(opt)

    #os.makedirs(opt.output_dir, exist_ok=True)

    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    test_dataset = ImageDataset(opt.input_dir, transforms_ = transforms_, mode = "test")
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False) 
    
    #Load model
    model = eval(opt.model_name)()
    model = model.cuda() 
    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for i, (img, path) in enumerate(test_dataloader):
        name = path[0].split('/')[-1].split('.')[0]
        
        #to cuda
        img = img.cuda()
        
        #forward
        output = model(img)

        #predicted class
        predicted_class = torch.argmax(output, axis = 1)
        predicted_class = predicted_class.cpu().squeeze()

        #to rgb
        predicted_img = decode_label(predicted_class)

        #write to file
        output_path = os.path.join(os.path.join(opt.output_dir, f"{name}.png"))
        save_image(predicted_img, output_path)

print(f"save to {opt.output_dir}")