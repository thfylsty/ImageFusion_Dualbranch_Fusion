# -*- coding: utf-8 -*-
from activate_fuction import new
import torch.nn as nn
import torch
Activate = new  # replace nn.LeakyReLU

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv2', nn.Conv2d(128,64,3,1,1))
        self.layers.add_module('Act2' , Activate(inplace=True))
        self.layers.add_module('Conv3', nn.Conv2d(64,32,3,1,1))
        self.layers.add_module('Act3' , Activate(inplace=True))
        self.layers.add_module('Conv4', nn.Conv2d(32,16,3,1,1))
        self.layers.add_module('Act4' , Activate(inplace=True))
        self.layers.add_module('Conv5', nn.Conv2d(16,1,3,1,1))

    def forward(self, x):
        return self.layers(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.Activate  = Activate(inplace=True)


        self.Conv_d = nn.Conv2d(32, 16, 3, 1, 1)
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16,16,3,1,1),
            'DenseConv2': nn.Conv2d(32,16,3,1,1),
            'DenseConv3': nn.Conv2d(48,16,3,1,1)
        })

        self.Conv2 = nn.Conv2d(32, 64, 3,2,1)
        self.Conv3 = nn.Conv2d(64, 128, 3,2,1)
        self.Conv4 = nn.Conv2d(128, 64, 3,2,1)
        self.Upsample = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        # self.SELayer = SELayer(128,8)

    def con(self,x,isTest = False):
        # print(x.shape)
        x = self.Activate(self.Conv1(x))
        # x = self.Activate(self.Conv1_2(x))
        x_d = self.Activate(self.Conv_d(x))
        x_s = x
        for i in range(len(self.layers)):
            out = self.Activate(self.layers['DenseConv' + str(i + 1)](x_d))
            x_d = torch.cat([x_d, out], 1)
        # out = x
        x_s = self.Activate(self.Conv2(x_s))
        x_s = self.Activate(self.Conv3(x_s))
        x_s = self.Activate(self.Conv4(x_s))
        x_s = self.Upsample(x_s)

        if isTest: # Upsample for Test, The size of some pictures is not a power of 2
            # print('test')
            test_upsample = nn.Upsample(size=(x.shape[2],x.shape[3]),mode='bilinear',align_corners=True)
            x_s = test_upsample(x_s)

        out = torch.cat([x_d, x_s], 1)
        # out = self.SELayer(out)
        return out

    def forward(self, x,isTest=False):
        x = self.con(x,isTest=isTest)
        return x


class DenseFuseNet(nn.Module):
    
    def __init__(self):
        super(DenseFuseNet,self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,x,isTest=False,saveMat=False):
        x = self.encoder(x,isTest=isTest)
        out = self.decoder(x)
        return out

