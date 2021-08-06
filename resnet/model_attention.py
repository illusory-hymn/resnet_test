import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import cv2
from skimage import measure
from utils.func import *

class SE(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction,channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        h, w, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return  y

class  GC(nn.Module):
    def __init__(self, channel, reduction=8):
        super(GC, self).__init__()
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.LayerNorm([channel // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            )
        self.conv_mask = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        batch, channel, h, w = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, h*w).unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, h*w)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)

        channel_add_term = self.channel_add_conv(context)
        out =  channel_add_term
        return out


class Model(nn.Module): 
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_classes = args.num_classes 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  ## -> 64x224x224
        self.relu1_1 = nn.ReLU(inplace=True) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) ## -> 64x224x224
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  ## -> 64x112x112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  ## -> 128x112x112
        self.relu2_1 = nn.ReLU(inplace=True) 
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) ## -> 128x112x112
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2) ## -> 128x56x56
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1) ## -> 256x56x56
        self.relu3_1 = nn.ReLU(inplace=True) 
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1) ## -> 256x56x56
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1) ## -> 256x56x56
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2) ## -> 256x28x28 

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1) ## -> 512x28x28
        self.relu4_1 = nn.ReLU(inplace=True) 
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x28x28
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x28x28
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2) ## -> 512x14x14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu5_1 = nn.ReLU(inplace=True) 
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2) ## -> 512x14x14

        self.conv_copy_5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu_copy_5_1 = nn.ReLU(inplace=True) 
        self.conv_copy_5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu_copy_5_2 = nn.ReLU(inplace=True)
        self.conv_copy_5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu_copy_5_3 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(14) ## ->(512,1,1)
        self.saliency_max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.saliency_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.classifier_loc = nn.Sequential( 
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )  
        self.classifier_loc2 = nn.Sequential( 
            nn.Conv2d(512, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )  
        self.classifier_cls_copy = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, label=None):
        ##  change weight
        self.change_weight()
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        #self.show_func(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        #self.show_func(x)
        x_4 = x.clone() 
        
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        #self.show_func(x)
        x_5 = x.clone()

        x = self.classifier_cls(x)
        batch, channel, h, w = x.size()
        self.x_sum = torch.zeros(batch, 1, h, w).cuda()
        if label !=None:
            for i in range(batch):
                self.x_sum[i][0] = x[i][label[i]]
        #self.show_func(x)
        #self.show_func(self.x_sum)
        self.feature_map = x
        x_6 = x.clone()

        x_saliency = self.classifier_loc(x_4)
        x_saliency = x_saliency.mean(1).unsqueeze(1)
        x_saliency = (x_saliency - 0.5 ) *2
        self.x_saliency = x_saliency
        self.x_saliency_avg = self.saliency_avg_pool(x_saliency)

        x_2_saliency = self.classifier_loc2(x_4)
        x_2_saliency = x_2_saliency.mean(1).unsqueeze(1)
        x_2_saliency = (x_2_saliency - 0.5 ) *2
        self.x_2_saliency = x_2_saliency

## erase 
        x_saliency_erase = 1 - x_saliency
        x_erase = x_4.detach() * x_saliency_erase
        #self.show_func(x_erase)
        x_erase = self.pool4(x_erase)
        x_erase = self.conv_copy_5_1(x_erase)
        x_erase = self.relu_copy_5_1(x_erase)
        x_erase = self.conv_copy_5_2(x_erase)
        x_erase = self.relu_copy_5_2(x_erase)
        x_erase = self.conv_copy_5_3(x_erase)
        x_erase = self.relu_copy_5_3(x_erase)
        #self.show_func(x_erase)
        x_erase = self.classifier_cls_copy(x_erase)
        self.x_erase_sum = torch.zeros(batch, 1, h, w).cuda()
        if label !=None:
            for i in range(batch):
                self.x_erase_sum[i][0] = x_erase[i][label[i]]
        
## score
        x = x_6 
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_1 = x

        return self.score_1
    
    def change_weight(self):
        self.conv_copy_5_1.weight.data = self.conv5_1.weight.data
        self.conv_copy_5_1.bias.data = self.conv5_1.bias.data
        self.conv_copy_5_2.weight.data = self.conv5_2.weight.data
        self.conv_copy_5_2.bias.data = self.conv5_2.bias.data
        self.conv_copy_5_3.weight.data = self.conv5_3.weight.data
        self.conv_copy_5_3.bias.data = self.conv5_3.bias.data

        self.classifier_cls_copy[0].bias.data = self.classifier_cls[0].bias.data 
        self.classifier_cls_copy[0].weight.data = self.classifier_cls[0].weight.data 
        self.classifier_cls_copy[2].bias.data = self.classifier_cls[2].bias.data 
        self.classifier_cls_copy[2].weight.data = self.classifier_cls[2].weight.data 
        self.classifier_cls_copy[4].bias.data = self.classifier_cls[4].bias.data 
        self.classifier_cls_copy[4].weight.data = self.classifier_cls[4].weight.data 
        
    def show_func(self, feature_map):
        feature = feature_map.clone().detach()
        img = feature.mean(1)
        img = np.array(img.data.cpu())[0]
        max_val = np.max(img, axis=(0,1))
        min_val = np.min(img, axis=(0,1))
        img = (img -min_val) / (max_val - min_val+1e-5)
        img = cv2.resize(img, (224,224))
        plt.imshow(img)
        plt.axis('off')
        #plt.savefig('output/' + img_path)
        plt.show()

    def loss_fg_func(self, x, y, conf=1):
        #x = nn.Sigmoid()(x)
        x = x * (1 - 2e-6) + 1e-6
        loss = -(y * torch.log(x) + (1-y) * torch.log(1-x)) 
        loss = loss.mean(0)
        return loss

    def loss_bg_func(self, x, y, conf=1):
        #x = nn.Sigmoid()(x)
        x = x * (1 - 2e-6) + 1e-6
        loss = -(y * torch.log(x) + (1-y) * torch.log(1-x))
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
    #def get_loss_1(self):
    #    return self.seed_loss(self.x_saliency, self.proposal, self.back, 0.5 , 0.5, 1, 1) 

    '''def get_loss_1(self):
        batch = self.score_1.size(0)
        x_saliency = self.x_saliency.clone().view(batch,-1)
        propsal = self.back.clone().detach().view(batch,-1)
        loss = (x_saliency - propsal) * (x_saliency - propsal) 
        loss = torch.mean(loss, -1)
        loss = loss.mean(0)
        return loss'''
    
    def get_loss_2(self):    
        return self.seed_loss(self.back, self.back, self.back, 0.6, 0.6, 1,1)

    def get_loss_3(self):
        return self.seed_loss(self.x_2_saliency, self.x_saliency, self.x_saliency, 0.7, 0.7, 1,1) 
    
        

    '''def get_loss_4(self):
        batch, channel, h, w= self.back.size()
        back = self.back.clone().detach().view(batch, -1)
        back_res = self.back_res.clone().detach().view(batch, -1)
        x_saliency = self.x_saliency.clone()
        #x_saliency = self.normalize_atten_maps(x_saliency)
        x_2_saliency = x_saliency.view(batch, -1)
        #x_2_saliency = self.normalize_atten_maps(self.x_saliency.clone())
        #back_score = self.classifier_linear(back_sum) *28
        fg = x_2_saliency * back 
        fg = fg.mean(-1).unsqueeze(-1) 
        fg_score = fg
        bg = back_res * back 
        bg = bg.mean(-1).unsqueeze(-1)
        bg_score = bg
        score = torch.cat([fg_score, bg_score],1) * 28

        ##  cross_entropy
        score = torch.exp(score)
        score = score / (score.sum(dim=1, keepdim=True) )
        score = score[:,0]
        loss = torch.log(score)
        loss = - loss.mean(0)
        #print(self.classifier_linear.weight,self.classifier_linear.bias)
        return loss'''

    def get_loss_4(self):
        batch, channel, h, w= self.x_sum.size()
        x_sum = self.x_sum.clone().detach().view(batch, -1)
        x_erase_sum = self.x_erase_sum.clone().view(batch, -1)
        x_res = x_erase_sum
        x_res[x_res>x_sum] = 0
        x_res = x_res.sum(1)
        x_sum = x_sum.sum(1)
        x_res = x_res / (x_sum+1e-5)
        x_saliency = self.x_saliency.clone().view(batch, -1)
        x_saliency = x_saliency.sum(1) / (28*28)
        loss = x_res  + x_saliency 
        #loss = torch.clamp(loss, min=0)
        loss = loss.mean(0)
        return loss
       
        

    
    def iou (self, A, B):
        batch, channel, h, w = A.size()
        AB = A * B
        AB = torch.mean(AB.view(batch, channel,-1), -1)
        area =   A + B
        area = torch.mean(area.view(batch, channel,-1), -1)
        iou = AB / (area - AB + 1e-5)

        iou_mean = torch.mean(iou.view(batch, -1), -1).unsqueeze(-1)
        iou = iou / (iou_mean + 1e-5)
        iou = iou.unsqueeze(-1).unsqueeze(-1)
        return iou


    def max_feature(self, feature, top_num):
        batch, channel, h, w = feature.size()
        feature_map = feature.clone()
        weight = nn.MaxPool2d((h,w))(feature_map).squeeze(-1).squeeze(-1)
        value, target = weight.topk(top_num, 1, True, True)
        print(value)
        print(target)


    
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
        
def weight_init(m):
    classname = m.__class__.__name__  ## __class__将实例指向类，然后调用类的__name__属性
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
    elif classname.find('Linear') != -1:
       # m.weight.data.normal_(0, 0.01)
        m.weight.data.fill_(28)
        #m.bias.data.fill_(1.5)    

def model(args, pretrained=True):
    # base and dilation can be modified according to your needs
    model = Model(args)
    model.apply(weight_init)  #权值初始化
    pretrained_dict = torch.load('vgg16.pth')## 获得键名，例如：features.0.weight
    model_dict = model.state_dict() ## 存储网络结构名字和对应的参数(字典)
    model_conv_name = []

    for i, (k, v) in enumerate(model_dict.items()):
        model_conv_name.append(k)
    ii = 0
    for i, (k, v) in enumerate(pretrained_dict.items()):
        if k.split('.')[0] != 'features':
            break
        if np.shape(model_dict[model_conv_name[ii]]) == np.shape(v):
            model_dict[model_conv_name[ii]] = v 
            if i >= 20 and i <=25:
                model_dict[model_conv_name[ii+6]] = v 
            if i == 25:
                ii = ii + 6
            ii = ii +1
    model.load_state_dict(model_dict)
    print("pretrained weight load complete..")
    return model
