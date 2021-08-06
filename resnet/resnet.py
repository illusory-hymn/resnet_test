"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from util import remove_layer
from util import replace_layer
from util import initialize_weights
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCam(nn.Module):
    def __init__(self, block, layers, args, num_classes=1000,
                 large_feature_map=True):
        super(ResNetCam, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.classifier_loc = nn.Sequential( 
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )  
        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, label=None, return_cam=False):
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)
        layer4_copy = copy.deepcopy(self.layer4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x_2 = x.clone()
        #self.show_func(x)
        x = self.layer3(x)
        x_3 = x.clone()
        #self.show_func(x)
        x_saliency = self.classifier_loc(x_3)
        x_saliency = x_saliency.mean(1).unsqueeze(1)
        x_saliency = (x_saliency - 0.5 ) *2
        self.x_saliency = x_saliency
        
        x = F.max_pool2d(x, kernel_size=2)
        x = self.layer4(x)
        x_4 = x.clone()
        
        x = self.classifier_cls(x)
        self.feature_map = x
        batch, channel, h, w = x_4.size()
        self.x_sum = torch.zeros(batch, 1, h, w).cuda()
        for i in range(batch):
            self.x_sum[i][0] = x[i][label[i]]

        self.score_1 = self.avgpool(x).squeeze(-1).squeeze(-1)

##  erase
        x_saliency_erase = 1 - x_saliency
        x_erase = x_3.detach() * x_saliency_erase
        #self.show_func(x_erase)
        x_erase = F.max_pool2d(x_erase, kernel_size=2)
        x_erase = layer4_copy(x_erase)
        #self.show_func(x_erase)
        x_erase = classifier_cls_copy(x_erase)

        self.x_erase_sum = torch.zeros(batch, 1, h, w).cuda()
        for i in range(batch):
            self.x_erase_sum[i][0] = x_erase[i][label[i]]

## x_3_atten
        #x_3 = self.classifier_loc[0](x_3)
        #x_3 = self.classifier_loc[1](x_3)
        #x_3 = self.classifier_loc[2](x_3)
        #x_3 = self.classifier_loc[3](x_3)
        '''channel = x_3.size(1)
        x_3_normal = self.normalize_atten_maps(x_3.clone())
        x_3_normal_sum = x_3_normal.view(batch,channel,-1).sum(-1)
        x_saliency_sum = x_saliency.clone().detach().view(batch,1,-1).sum(-1)
        atten = self.x_saliency * x_3_normal
        atten = atten.view(batch, channel,-1).sum(-1)
        atten = atten / (x_3_normal_sum  +1e-5)
        atten = atten.unsqueeze(-1).unsqueeze(-1)
        back_atten = atten - (1-atten)
        x_3_atten = back_atten *  x_3
        x_3_atten[x_3_atten<0] = 0
        self.x_3_atten = x_3_atten.mean(1).unsqueeze(1)
        self.show_func(x_3_atten)'''
       
       
        return self.score_1

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers
        

    def show_func(self, feature_map):
        feature = feature_map.clone().detach()
        img = feature.mean(1)
        img = np.array(img.data.cpu())[0]
        max_val = np.max(img, axis=(0,1))
        min_val = np.min(img, axis=(0,1))
        img = (img -min_val) / (max_val - min_val)
        img = cv2.resize(img, (224,224))
        plt.imshow(img)
        plt.axis('off')
        #plt.savefig('output/' + img_path)
        plt.show()

    def loss_fg_func(self, x, y, conf=1):
        #x = nn.Sigmoid()(x)
        x = x * (1 - 2e-6) + 1e-6
        loss = -(y * torch.log(x) + (1-y) * torch.log(1-x)) * conf
        loss = loss.mean(0)
        return loss

    def loss_bg_func(self, x, y, conf=1):
        #x = nn.Sigmoid()(x)
        x = x * (1 - 2e-6) + 1e-6
        loss = -(y * torch.log(x) + (1-y) * torch.log(1-x)) * conf
        loss = loss.mean(0)
        return loss

    def seed_loss(self, x_target, x_source , x_background, target_thr, back_thr, h_1=1, h_2=1):
## target_loss
        source_seed = x_source.clone().detach() 
        source_seed = self.normalize_atten_maps(source_seed)
        source_seed[source_seed>target_thr] = 1
        source_seed[source_seed<1] = 0

        '''source_seed_x = torch.max(source_seed, 2)[0].unsqueeze(2)
        source_seed_y = torch.max(source_seed, -1)[0].unsqueeze(-1)
        source_seed_xy = source_seed_y * source_seed_x
        source_seed_x_y = source_seed_y * source_seed + source_seed_x * source_seed
        source_seed_x_y[source_seed_x_y>1] = 1
        return source_seed'''
        source_seed = source_seed.view(-1,1)
        source_seed_position = source_seed == 1

        target = x_target.clone()
        #target = self.normalize_atten_maps(target)
        target = target.view(-1,1)
        

        if torch.sum(source_seed_position)>0:
            loss_target = self.loss_fg_func(target[source_seed_position], source_seed[source_seed_position]) 
        else:
            loss_target = torch.tensor(0,dtype=float).cuda()
            
## back_loss
        back_seed = x_background.clone().detach() 
        back_seed = self.normalize_atten_maps(back_seed) ## 使每次都产生background
        back_seed[back_seed>back_thr] = 1
        back_seed[back_seed<1] = 0
        #return back_seed 
        back_seed = Variable(back_seed).view(-1,1) ## 0表示背景
        back_seed_position = back_seed == 0

        if torch.sum(back_seed_position)>0:
            loss_background = self.loss_bg_func(target[back_seed_position], back_seed[back_seed_position]) ## 只考虑背景损失，也就是只在background=1时计算损失
        else:
            loss_background = torch.tensor(0,dtype=float).cuda()

        return loss_target * h_1 + loss_background *h_2

    def get_loss_3(self):
        return self.seed_loss(self.x_2_saliency, self.x_saliency, self.x_saliency, 0.7, 0.7, 1,1) 
    

    def get_loss_4(self):
        batch, channel, h, w= self.x_sum.size()
        x_sum = self.x_sum.clone().detach().view(batch, -1)
        x_erase_sum = self.x_erase_sum.clone().view(batch, -1)
        x_res = x_erase_sum
       # x_res[x_res>x_sum] = 0
        x_res = x_res.sum(1)
        x_sum = x_sum.sum(1)
        x_res = x_res / (x_sum+1e-5)
        x_saliency = self.x_saliency.clone().view(batch, -1)
        x_saliency = x_saliency.sum(1) / (28*28)
        loss = x_res * 0.8 + x_saliency
        #loss = torch.clamp(loss, min=0)
        loss = loss.mean(0)
        return loss

    
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed
    
    def get_fused_cam(self, target=None):
        batch, channel, _, _ = self.feature_map.size()

        if target is None:
            _, target = self.score_1.topk(1, 1, True, True)
            target = target.squeeze() ## torch.topk沿dim返回前k个最大值和位置（和max一样）
        
        cam = self.feature_map[0][target].unsqueeze(0)
        cam = cam 
        return cam    


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

def load_pretrained_model(model):
    strict_rule = True

    state_dict = torch.load('resnet50-19c8e357.pth')

    state_dict = remove_layer(state_dict, 'fc')
    strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def model(args, pretrained=True):
    model = ResNetCam(Bottleneck, [3, 4, 6, 3], args)
    if pretrained:
        model = load_pretrained_model(model)
    return model
