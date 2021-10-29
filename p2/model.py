# The code below is modified from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
import torch
import torch.nn as nn
from torchvision import models

class Vgg16_bn(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_bn, self).__init__()
        vgg_pretrained_features = models.vgg16_bn(pretrained=True).features
        print(vgg_pretrained_features)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(24):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 34):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(34, 44):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        out = [h_relu1, h_relu2, h_relu3]
        return out

class FCN_32(torch.nn.Module):
    def __init__(self, num_classes = 7, requires_grad=False):
        super(FCN_32, self).__init__()
        self.vgg16_bn = Vgg16_bn(requires_grad)
        self.fconv = torch.nn.Sequential(torch.nn.Conv2d(512, num_classes, 1), 
                                        torch.nn.ReLU())
        #self.upsampling_block = torch.nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 32, padding = 1, dilation= 11)
        self.upsampling_block = torch.nn.Upsample(scale_factor = 32, mode="bilinear")
    def forward(self, x):
        out = self.vgg16_bn(x)
        feat = self.fconv(out[-1])
        score = self.upsampling_block(feat)
        return score

class FCN_16(torch.nn.Module):
    def __init__(self, num_classes = 7, requires_grad=False):

        super(FCN_16, self).__init__()
        self.vgg16_bn = Vgg16_bn(requires_grad)
        self.fconv1 = torch.nn.Sequential(torch.nn.Conv2d(512, num_classes, 1), 
                                        torch.nn.ReLU())
        self.fconv2 = torch.nn.Sequential(torch.nn.Conv2d(512, num_classes, 1), 
                                        torch.nn.ReLU())

        #self.upsampling_block = torch.nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 16, padding = 0, dilation= 5)
        self.upsampling_block = torch.nn.Upsample(scale_factor = 16, mode="bilinear")
    def forward(self, x):

        out = self.vgg16_bn(x)
        feat1 = self.fconv1(out[1])

        feat2 = self.fconv2(out[2])
        feat2 = torch.nn.Upsample(scale_factor = 2, mode = "bilinear")(feat2)

        combine = feat1 + feat2

        score = self.upsampling_block(combine)
        return score

class FCN_8(torch.nn.Module):
    def __init__(self, num_classes = 7, requires_grad=False):

        super(FCN_8, self).__init__()
        self.vgg16_bn = Vgg16_bn(requires_grad)
        self.fconv0 = torch.nn.Sequential(torch.nn.Conv2d(256, num_classes, 1), 
                                        torch.nn.ReLU())
        self.fconv1 = torch.nn.Sequential(torch.nn.Conv2d(512, num_classes, 1), 
                                        torch.nn.ReLU())
        self.fconv2 = torch.nn.Sequential(torch.nn.Conv2d(512, num_classes, 1), 
                                        torch.nn.ReLU())

        #self.upsampling_block = torch.nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 16, padding = 0, dilation= 5)
        self.upsampling_block = torch.nn.Upsample(scale_factor = 8, mode="bilinear")
    def forward(self, x):
        out = self.vgg16_bn(x)

        feat0 = self.fconv0(out[0])

        feat1 = self.fconv1(out[1])
        feat1 = torch.nn.Upsample(scale_factor = 2, mode = "bilinear")(feat1)

        feat2 = self.fconv2(out[2])
        feat2 = torch.nn.Upsample(scale_factor = 4, mode = "bilinear")(feat2)

        combine = feat0 + feat1 + feat2

        score = self.upsampling_block(combine)
        return score


class ResNet_50(torch.nn.Module):
    def __init__(self, requires_grad = False):

        super(ResNet_50, self).__init__()
        self.resnet_50 = models.resnet50(pretrained = True)

        self.slice1 = torch.nn.Sequential(self.resnet_50.conv1, 
                                          self.resnet_50.bn1, 
                                          self.resnet_50.relu, 
                                          self.resnet_50.maxpool, 
                                          self.resnet_50.layer1)

        self.slice2 = torch.nn.Sequential(self.resnet_50.layer2)
        self.slice3 = torch.nn.Sequential(self.resnet_50.layer3)
        self.slice4 = torch.nn.Sequential(self.resnet_50.layer4)

    def forward(self, x):
        relu_1 = self.slice1(x)
        relu_2 = self.slice2(relu_1)
        relu_3 = self.slice3(relu_2)
        relu_4 = self.slice4(relu_3)

        return [relu_2, relu_3, relu_4]

class FCN_resnet50_8(torch.nn.Module):
    def __init__(self, num_classes = 7, requires_grad=False):

        super(FCN_resnet50_8, self).__init__()
        self.resnet_50= ResNet_50(requires_grad)
        self.fconv0 = torch.nn.Sequential(torch.nn.Conv2d(512, num_classes, 1), 
                                        torch.nn.ReLU())
        self.fconv1 = torch.nn.Sequential(torch.nn.Conv2d(1024, num_classes, 1), 
                                        torch.nn.ReLU())
        self.fconv2 = torch.nn.Sequential(torch.nn.Conv2d(2048, num_classes, 1), 
                                        torch.nn.ReLU())

        #self.upsampling_block = torch.nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 16, padding = 0, dilation= 5)
        self.upsampling_block = torch.nn.Upsample(scale_factor = 8, mode="bilinear")
    def forward(self, x):
        out = self.resnet_50(x)

        feat0 = self.fconv0(out[0])

        feat1 = self.fconv1(out[1])
        feat1 = torch.nn.Upsample(scale_factor = 2, mode = "bilinear")(feat1)

        feat2 = self.fconv2(out[2])
        feat2 = torch.nn.Upsample(scale_factor = 4, mode = "bilinear")(feat2)

        combine = feat0 + feat1 + feat2

        score = self.upsampling_block(combine)
        return score


#resnet50 = FCN_resnet50_8()
#x = torch.randn(16, 3, 512, 512)
#output = resnet50(x)
#import pdb; pdb.set_trace()