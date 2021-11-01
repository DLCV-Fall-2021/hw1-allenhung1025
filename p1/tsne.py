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
import csv
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../hw1_data/p1_data/val_50", help="input directory")
    parser.add_argument("--output_tsne", type=str, default="../hw1_data/p1_data/tsne.png", help="output csv")
    parser.add_argument("--model_path", type=str, default="./checkpoint_vggbn_224_deepest_feature_linear_scheduler_finetune/model_10.pth", help="model path")
    parser.add_argument("--model_name", type=str, default="Vgg19_bn_linear", help="name of the dataset")

    opt = parser.parse_args()
    print(opt)
    
    #transforms
    test_transforms_ = [

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]

    #load dataset
    test_dataset = ImageDataset(opt.input_dir, test_transforms_, mode="test") 

    #load dataloader
    test_dataloader = DataLoader(test_dataset, batch_size = 1)

    #load model
    model = eval(opt.model_name)(False, False)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.model_path)["model_state_dict"])
    model.eval()

    #csv file
    #csv_file = open(opt.output_csv, 'w', newline='')
    #csv_writer = csv.writer(csv_file) 
    #csv_writer.writerow(["image_id", "label"])

    #inference
    #accuracy = 0.0
    #embedding
    embedding_array = np.zeros((2500, 25088))
    predict_class_array = np.zeros((2500, ))

    #predicted class
    for i, (img, file_path) in enumerate(test_dataloader):
        img = img.cuda()
        output, second_last = model(img)
        embedding_array[i, :] = second_last.cpu().numpy()

        name = file_path[0].split('/')[-1]
        #label = int(name.split('_')[0])

        predict_class = output.argmax(axis = 1)
        predict_class = predict_class.item()
        predict_class_array[i] = predict_class
        #print(predict_class)
        #accuracy += (predict_class.item() == label)

        #csv_writer.writerow([str(name) , str(predict_class)])

    #print("acc: ", accuracy / len(test_dataset))

    ### tsne
    #plotting
    #code comes from tutorial https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

    tsne = TSNE(n_components=2,)
    feat_cols = [ 'pixel'+str(i) for i in range(embedding_array.shape[1]) ]
    df = pd.DataFrame(embedding_array,columns=feat_cols)
    df['y'] = predict_class_array 
    df['label'] = df['y'].apply(lambda i: str(i))

    data = df[feat_cols].values
    rndperm = np.random.permutation(2500)
    tsne_result = tsne.fit_transform(data)
    df["tsne-2d-one"] = tsne_result[:, 0]
    df["tsne-2d-two"] = tsne_result[:, 1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 50),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )
    plt.savefig(opt.output_tsne)