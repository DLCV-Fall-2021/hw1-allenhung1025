import argparse
import os
## The code is modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py
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

from model import Vgg19, Vgg19_linear, Vgg19_bn, Vgg19_bn_linear, resnext101_32x8d_linear
from dataset import ImageDataset

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
#from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--root_train", type=str, default="../hw1_data/p1_data/train_50", help="name of the dataset")
    parser.add_argument("--root_val", type=str, default="../hw1_data/p1_data/val_50", help="name of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--val_batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint_vgg_linear", help="checkpoint directory")
    parser.add_argument("--requires_grad", action = "store_true")
    parser.add_argument("--model_name", type=str, default="Vgg_linear", help="model name")
    opt = parser.parse_args()
    print(opt)

    #miscellaneous
    x_train_loss = [x + 1 for x in range(opt.n_epochs)]
    x_val_loss = [x + opt.checkpoint_interval for x in range(0, opt.n_epochs, opt.checkpoint_interval)]
    y_train_loss = []
    y_val_loss = []

    y_train_acc = []
    y_val_acc = []
    #plt.plot(x_1, y_1, label="train loss")
    #plt.plot(x_2, y_2, label="val loss")
    #plt.legend()
    #plt.savefig("./test.png")

    #make checkpoint dir
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    #transforms
    train_transforms_ = [

        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p = 0.5),
        #transforms.RandomRotation((0, 180)),
        #transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
    ]

    val_transforms_ = [

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]

    #training and validation dataset
    train_dataset = ImageDataset(opt.root_train, train_transforms_, mode="train")
    val_dataset = ImageDataset(opt.root_val, val_transforms_, mode="val")

    #training and validation loader
    train_loader = DataLoader(train_dataset, batch_size = opt.train_batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size = opt.val_batch_size, shuffle = True, drop_last = True)

    #Load model
    if opt.model_name == "resnext":
        model = resnext101_32x8d_linear() 
        model = model.cuda()
    else:
        model = eval(opt.model_name)(opt.requires_grad)
        model = model.cuda()

    #Criterion
    cross_entropy_loss = nn.CrossEntropyLoss()

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas = (opt.b1, opt.b2), lr = opt.lr)

    #Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma = 0.5)

    for ep in range(opt.n_epochs):

        loss = 0.0
        train_acc = 0.0

        #model train
        model.train()

        for i, (data, label) in enumerate(train_loader):
            ##to cuda
            data = data.cuda()
            label = label.cuda()

            ##forward
            output, _ = model(data)

            ##compute loss
            batch_loss = cross_entropy_loss(output, label)
            loss += batch_loss

            ##compute accuracy
            predict_class = torch.argmax(output, axis=1)
            same_label = (predict_class == label)
            correct_num = same_label.sum().item()
            train_acc += correct_num

            ##zero grad
            optimizer.zero_grad()

            ##backward
            batch_loss.backward()
            #print(batch_loss.item())

            #step
            optimizer.step()

        loss /= len(train_loader)
        y_train_loss.append(loss.item())

        train_acc /= (len(train_loader) * opt.train_batch_size)
        y_train_acc.append(train_acc)

        print("train: ", train_acc)
        #update scheduler
        scheduler.step()

        if (ep + 1) % opt.checkpoint_interval == 0:
            ## model to eval
            model.eval()

            ##validation
            val_loss = 0.0
            val_acc = 0.0

            for j, (val_data, val_label) in enumerate(val_loader):
                ### to cuda
                val_data = val_data.cuda()
                val_label = val_label.cuda()

                ###forward
                val_output, _ = model(val_data)

                ###compute loss
                val_batch_loss = cross_entropy_loss(val_output, val_label)
                val_loss += val_batch_loss.item()

                ###compute accuracy
                val_predict_class = torch.argmax(val_output, axis=1)
                val_same_label = (val_predict_class == val_label)
                val_correct_num = val_same_label.sum().item()
                val_acc += val_correct_num

            val_loss /= len(val_loader)
            y_val_loss.append(val_loss)

            val_acc /= (len(val_loader) * opt.val_batch_size)
            y_val_acc.append(val_acc)
            print("val: ", val_acc)

            if (ep + 1) == opt.checkpoint_interval:
                best_val_loss = val_loss
                best_train_loss = loss

                best_val_acc = val_acc
                best_train_acc = train_acc

                best_model = model
                best_epoch = ep
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_train_loss = loss

                    best_val_acc = val_acc
                    best_train_acc = train_acc

                    best_model = model
                    best_epoch = ep

            ##save interval checkpoint
            torch.save({ "model_state_dict": model.state_dict(),
                         "epoch": ep + 1,
                         "train_loss": loss,
                         "train_acc": train_acc,
                         "val_loss": val_loss,
                         "val_acc": val_acc
                        }, f"{opt.checkpoint_dir}/model_{ep + 1}.pth")

    #save the best checkpoint
    torch.save({ "model_state_dict": best_model.state_dict(),
                 "epoch": best_epoch + 1,
                 "train_loss": best_train_loss,
                 "train_acc": best_train_acc,
                 "val_loss": best_val_loss,
                 "val_acc": best_val_acc
                }, f"{opt.checkpoint_dir}/model_best.pth")


    #Mischellaneous
    fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)

    ax1.plot(x_train_loss, y_train_loss, label = "train loss")
    ax1.plot(x_val_loss, y_val_loss, label = "val loss")
    ax1.legend()
    ax1.set_title("cross entropy loss")

    ax2.plot(x_train_loss, y_train_acc, label = "train accuracy")
    ax2.plot(x_val_loss, y_val_acc, label = "val accuracy")
    ax2.legend()
    ax2.set_title("accuracy")
    plt.savefig(f"./{opt.checkpoint_dir}/loss_acc.png")