import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import argparse
from ofa.utils import MyModule, MyNetwork, MyGlobalAvgPool2d, min_divisible_value, SEModule
from ofa.utils import get_same_padding, make_divisible, build_activation, init_models
from collections import OrderedDict

__all__ = ['ProgramNet', 'ProgramModule','ResProgramNet','MobiProgramNet']



class ProgramModule(MyNetwork):
    def __init__(self, input_size=224, in_channels=3, out_channels=3,
                    expand=1.0, kernel_size=5, act_func='relu', n_groups=2,
                    downsample_ratio=2, upsample_ratio=2, upsample_type='bilinear', stride=1):
        super(ProgramModule, self).__init__()
        self.input_size = input_size
        if downsample_ratio is not None:
            upsample_ratio = downsample_ratio//upsample_ratio
        if self.input_size == 7:
            kernel_size = 3
        self.encoder_config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'expand': expand,
            'kernel_size': kernel_size,
            'act_func': act_func,
            'n_groups': n_groups,
            'downsample_ratio': downsample_ratio,
            'upsample_type': upsample_type,
            'upsample_ratio': upsample_ratio,
            'stride': stride,
        }

        padding = get_same_padding(kernel_size)
        if downsample_ratio is None:
            pooling = nn.AvgPool2d(2, 2, 0)
        else:
            pooling = nn.AvgPool2d(downsample_ratio, downsample_ratio, 0)
        #only for resnet
        # expand = 1/4
        # if out_channels == 64:
        #     num_mid = 64
        # else:
        #     num_mid = make_divisible(int(in_channels * expand), divisor=MyNetwork.CHANNEL_DIVISIBLE)

        num_mid = make_divisible(int(in_channels * expand), divisor=MyNetwork.CHANNEL_DIVISIBLE)

        self.encoder = nn.Sequential(OrderedDict({
            'pooling': pooling,
            'conv1': nn.Conv2d(in_channels, num_mid, kernel_size, stride, padding, groups=n_groups, bias=False),
            'bn1': nn.BatchNorm2d(num_mid, eps=0.001),
            'act': build_activation(act_func),
            'conv2': nn.Conv2d(num_mid, out_channels, 1, 1, 0, bias=False),
            'final_bn': nn.BatchNorm2d(out_channels, eps=0.001),
        }))

        # initialize
        init_models(self.encoder)
        self.encoder.final_bn.weight.data.zero_()

    def forward(self, x):
        
        encoder_x = self.encoder(x)
        if self.encoder_config['upsample_ratio'] is not None:
            encoder_x = F.upsample(encoder_x, (x.shape[2]//self.encoder_config['upsample_ratio'], x.shape[3]//self.encoder_config['upsample_ratio']),
                                            mode=self.encoder_config['upsample_type'])
        return encoder_x

    @staticmethod
    def has_encoder_module(net):
        for m in net.modules():
            if isinstance(m, ProgramModule):
                return True
        return False
    

class ProgramNet(MyNetwork):
    def __init__(self, net, input_size, in_channels=3):
        super(ProgramNet, self).__init__()
        self.input_size = input_size        
        
        # option: adding encoder after pooling
        self.encoder_index = [0,1,5,9,13,17,21]

        self.input_size_lists = [self.input_size//2,self.input_size//4,self.input_size//8,self.input_size//16,self.input_size//16,self.input_size//32,self.input_size//32]     
        self.in_channel_lists = [3, 3, 32, 40, 80,96, 192]
        self.out_channel_lists = [16, 32, 40, 80, 96, 192, 320]

        sidemodules = []

        for i in self.encoder_index:
            if i == 0:
                continue
            elif i == 1:
                encodermodule = ProgramModule(self.input_size_lists[i], self.in_channel_lists[i], self.out_channel_lists[i], n_groups=1, downsample_ratio=4, upsample_ratio=2)
            # elif i == 5:
            #     idx = self.encoder_index.index(i)
            #     # encodermodule = EncoderModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx], n_groups=4, downsample_ratio=None)

            #     encodermodule = ProgramModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx], downsample_ratio=4, upsample_ratio=2)
            elif i in [5,9,17]: #reduce dimension
                idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx], downsample_ratio=None)
            else:
                idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx])
            sidemodules.append(encodermodule)
        self.sidemodules = nn.Sequential(*sidemodules)

        self.pool = nn.AvgPool2d(2, 2, 0)
        self.main_branch = net

    def forward(self, x):


        side_x = self.pool(x)
        side_x = self.sidemodules[0](side_x) 
        
        # detailed self.net
        x = self.main_branch.first_conv(x)
        count = 1
        for idx, block in enumerate(self.main_branch.blocks):
            
            if idx in self.encoder_index[1:]:
                if idx == 1:  # block 1
                    x = block(x)
                    x += side_x
                else:  # block 5,9,13...
                    main_x = block(x)
                    side_x = self.sidemodules[count](x)
                    x = side_x+main_x
                    count += 1
            else:
                x = block(x)

        if self.main_branch.feature_mix_layer is not None:
            x = self.main_branch.feature_mix_layer(x)
     
        x = self.main_branch.global_avg_pool(x)

        x = self.main_branch.classifier(x)

        return x
        

class ResProgramNet(MyNetwork):
    def __init__(self, net, input_size, in_channels=3):
        super(ResProgramNet, self).__init__()
        self.input_size = input_size
                
        # option: adding encoder after pooling
        self.encoder_index = [0,1,2,3,4,5]
        self.input_size_lists = [self.input_size//4,self.input_size//4,self.input_size//8,self.input_size//16,self.input_size//32,self.input_size//32]     

        self.in_channel_lists = [3, 64, 256, 512, 1024,2048]

        self.out_channel_lists = [64, 256, 512, 1024, 2048,2048]

        sidemodules = []

        for i in self.encoder_index :
            if i == 0:
                encodermodule = ProgramModule(self.input_size_lists[i], self.in_channel_lists[i], self.out_channel_lists[i],  n_groups=1)
            elif i in [2,3,4]: #reduce dimension
                # idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[i], self.in_channel_lists[i], self.out_channel_lists[i], downsample_ratio=None)
            else:
                # idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[i], self.in_channel_lists[i], self.out_channel_lists[i])
            sidemodules.append(encodermodule)
        self.sidemodules = nn.Sequential(*sidemodules)

        self.main_branch = net
        self.pool = nn.AvgPool2d(4, 4, 0)

    def forward(self, x):
        # x = self.reprogram(x)
        # x = self.main_branch(x)
        
        side_x = self.pool(x)
        side_x = self.sidemodules[0](side_x)
        x = self.main_branch.conv1(x)
        x = self.main_branch.bn1(x)
        x = self.main_branch.relu(x)
        x = self.main_branch.maxpool(x) + side_x

        for i in range(len(self.main_branch.layer1)):
            if i == 0:
                main_x = self.main_branch.layer1[i](x)
                side_x = self.sidemodules[1](x)
                x = main_x + side_x
                x+=side_x
            else:
                x = self.main_branch.layer1[i](x)
            # print(x.shape, side_x.shape)

        for i in range(len(self.main_branch.layer2)):
            if i == 0:
                main_x = self.main_branch.layer2[i](x)
                side_x = self.sidemodules[2](x)
                x = main_x + side_x
            else:
                x = self.main_branch.layer2[i](x)
            # print(x.shape)
        for i in range(len(self.main_branch.layer3)):
            if i == 0:
                main_x = self.main_branch.layer3[i](x)
                side_x = self.sidemodules[3](x)
                x = main_x + side_x
            else:                
                x = self.main_branch.layer3[i](x)
            # print(x.shape)
        for i in range(len(self.main_branch.layer4)):
            if i == 0:
                main_x = self.main_branch.layer4[i](x)
                side_x = self.sidemodules[4](x)
                x = main_x + side_x
            elif i == 2:
                main_x = self.main_branch.layer4[i](x)
                side_x = self.sidemodules[5](x)
                x = main_x + side_x                
            else:  
                x = self.main_branch.layer4[i](x)
            # print(x.shape)
        x = self.main_branch.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.main_branch.fc(x)
        

        return x


class MobiProgramNet(MyNetwork):
    def __init__(self,net_type, net, input_size, in_channels=3):
        super(MobiProgramNet, self).__init__()
        self.input_size = input_size
        # self.reprogram = ReProgramModule(input_size)
        # self.main_branch = net

        
        # option1: adding encoder after pooling
        if net_type == 'mobilenetv2':
            self.encoder_index = [2,4,7,11,14,17]
            self.input_size_lists = [self.input_size//4,self.input_size//4,self.input_size//8,self.input_size//16,self.input_size//16,self.input_size//32,self.input_size//32]     
            self.in_channel_lists = [3, 24, 32, 64,96,160]
            self.out_channel_lists = [24, 32, 64, 96,160,320]
        else:
            self.encoder_index = [2,4,7,11,13,15]
            self.input_size_lists = [self.input_size//4,self.input_size//8,self.input_size//16,self.input_size//16,self.input_size//32,self.input_size//32]     
            self.in_channel_lists = [3, 24, 40, 80,112,160]
            self.out_channel_lists = [24, 40, 80, 112,160,160]
            
        sidemodules = []

        for i in self.encoder_index :
            if i == 2:
                idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx],  n_groups=1)
            elif i in [4,7,13]: #reduce dimension
                idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx], downsample_ratio=None)
            else:
                idx = self.encoder_index.index(i)
                encodermodule = ProgramModule(self.input_size_lists[idx], self.in_channel_lists[idx], self.out_channel_lists[idx])
            sidemodules.append(encodermodule)
        self.sidemodules = nn.Sequential(*sidemodules)
        self.main_branch = net
        self.pool = nn.AvgPool2d(4, 4, 0)
        
    def forward(self, x):
        #x = self.reprogram(x)
        #x = self.main_branch(x)
        # this is for reprogram

        side_x = self.pool(x)
        side_x = self.sidemodules[0](side_x)


        count = 1
        for i in range(len(self.main_branch.features)):
            if i in self.encoder_index:
                if i == 2:
                    x = self.main_branch.features[i](x)
                    x += side_x
                else:
                    main_x = self.main_branch.features[i](x)
                    side_x = self.sidemodules[count](x)
                    x = side_x+main_x
                    count += 1
            else:  

                 x = self.main_branch.features[i](x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.main_branch.classifier(x)
        '''
        side_x = self.pool(x)
        side_x = self.sidemodules[0](side_x)
        x = self.main_branch.conv1(x)
        x = self.main_branch.bn1(x)
        x = self.main_branch.relu(x)
        x = self.main_branch.maxpool(x) + side_x

        for i in range(len(self.main_branch.layer1)):
            if i == 0:
                main_x = self.main_branch.layer1[i](x)
                side_x = self.sidemodules[1](x)
                x = main_x + side_x
                x+=side_x
            else:
                x = self.main_branch.layer1[i](x)
            # print(x.shape, side_x.shape)

        for i in range(len(self.main_branch.layer2)):
            if i == 0:
                main_x = self.main_branch.layer2[i](x)
                side_x = self.sidemodules[2](x)
                x = main_x + side_x
            else:
                x = self.main_branch.layer2[i](x)
            # print(x.shape)
        for i in range(len(self.main_branch.layer3)):
            if i == 0:
                main_x = self.main_branch.layer3[i](x)
                side_x = self.sidemodules[3](x)
                x = main_x + side_x
            else:                
                x = self.main_branch.layer3[i](x)
            # print(x.shape)
        for i in range(len(self.main_branch.layer4)):
            if i == 0:
                main_x = self.main_branch.layer4[i](x)
                side_x = self.sidemodules[4](x)
                x = main_x + side_x
            elif i == 2:
                main_x = self.main_branch.layer4[i](x)
                side_x = self.sidemodules[5](x)
                x = main_x + side_x                
            else:  
                x = self.main_branch.layer4[i](x)
            # print(x.shape)
        x = self.main_branch.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.main_branch.fc(x)
        '''

        return x