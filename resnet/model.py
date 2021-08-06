import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import cv2
from skimage import measure
from utils.func import *
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


        self.avg_pool = nn.AvgPool2d(14)  ## ->(512,1,1)
        self.saliency_max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.saliency_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 200, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.classifier_loc = nn.Sequential(
            nn.Conv2d(512,50, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ) 
        self.classifier_loc2 = nn.Sequential( 
            nn.Conv2d(512, 50, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )   
        self.classifier_loc3 = nn.Sequential( 
            nn.Conv2d(512, 50, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.classifier_loc4 = nn.Sequential( 
            nn.Conv2d(1, 1, kernel_size=1, padding=0),
        )
    def forward(self, x, label=None):
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
        x = self.pool3(x)

        x_3 = x.clone()
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x_4 = x.clone() 

        x = x_4
        x = self.pool4(x)

        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x_5 = x.clone()
 
        
        #self.show_func(x)
        x = self.classifier_cls(x)
        #self.show_func(x)
        self.feature_map = x
        x_6 = x.clone()

        x_saliency = self.classifier_loc(x_4)
        self.x_saliency = x_saliency.mean(1).unsqueeze(1)
        self.x_saliency = (self.x_saliency - 0.5 ) *2
        self.x_saliency_avg_pool = self.saliency_avg_pool(self.x_saliency)

        x_2_saliency = self.classifier_loc3(x_4)
        self.x_2_saliency = x_2_saliency.mean(1).unsqueeze(1)
        self.x_2_saliency = (self.x_2_saliency - 0.5 ) *2
        self.x_3_saliency = self.classifier_loc4(self.x_2_saliency)
        self.x_2_saliency_avg_pool = self.saliency_avg_pool(self.x_2_saliency)
        
        '''batch, channel, h, w = self.feature_map.size() 
        cam = torch.zeros((batch,1,h,w)).cuda()
        for i in range(batch):
            cam[i,0,:,:] = self.feature_map[i,label[i],:,:]
        cam =self.normalize_atten_maps(cam)

        conf_x = self.self_correlation(x_6.clone().detach(), cam)'''

        x = x_5.clone()
        x = nn.Upsample(size=(28,28), mode='bilinear',align_corners=False)(x)
       # x = self.normalize_atten_maps(x)
        back = self.classifier_loc2(x)
        self.back = back.mean(1).unsqueeze(1)
        self.back = (self.back - 0.5 ) *2
        self.back_avg_pool = self.saliency_avg_pool(self.back)
        
        self.proposal = (self.back.clone().detach() + self.x_saliency )/2
        

## score
        x = x_6.clone() * self.x_saliency_avg_pool.clone()
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_1 = x

        x = x_6.clone() * self.back_avg_pool.clone()
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_2 = x

        x = x_6.clone() * self.x_2_saliency_avg_pool.clone()
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_3 = x

        x = x_6.clone() 
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_4 = x

        batch = self.score_3.size(0)
        target_score = nn.Softmax(dim=1)(self.score_4)
        conf = torch.zeros((batch, 1)).cuda()
        if label != None :
            for i in range(batch):
                conf[i][0] = target_score[i][label[i]]
        self.conf = conf.unsqueeze(-1).unsqueeze(-1).expand_as(self.back).clone().detach()

        
        return self.score_1, self.score_2, self.score_3, self.score_4
    
    def show_func(self, feature_map):
        feature = feature_map.clone().detach()
        img = feature.mean(1)
        img = np.array(img.data.cpu())[0]
        max_val = np.max(img, axis=(0,1))
        min_val = np.min(img, axis=(0,1))
        img = (img -min_val) / (max_val - min_val)
        img = cv2.resize(img, (224,224))
        cv2.imshow('aa', img)
        #cv2.imwrite('output/'+img_path, img_add)
        cv2.waitKey(0)

    def loss_fg_func(self, x, y, conf):
        #x = nn.Sigmoid()(x)
        x = x * (1 - 2e-6) + 1e-6
        loss = -(y * torch.log(x) + (1-y) * torch.log(1-x))
        loss = loss.mean(0)
        return loss

    def loss_bg_func(self, x, y, conf):
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
        #return source_seed
        source_seed = source_seed.view(-1,1)
        source_seed_position = source_seed == 1

        target = x_target.clone()
        #target = self.normalize_atten_maps(target)
        target = target.view(-1,1)

        conf = self.conf
        conf = conf.view(-1,1)

        if torch.sum(source_seed_position)>0:
            loss_target = self.loss_fg_func(target[source_seed_position], source_seed[source_seed_position], conf[source_seed_position]) 
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
            loss_background = self.loss_bg_func(target[back_seed_position], back_seed[back_seed_position], conf[back_seed_position]) ## 只考虑背景损失，也就是只在background=1时计算损失
        else:
            loss_background = torch.tensor(0,dtype=float).cuda()

        return loss_target * h_1 + loss_background *h_2
    
    '''def get_loss_1(self): 
        return self.seed_loss(self.x_saliency, self.proposal, self.proposal, 0.5 , 0.5, 1, 1) '''
    def get_loss_1(self):
        batch = self.score_1.size(0)
        conf = self.conf
        x_saliency = self.x_saliency.clone() .view(batch, -1) 
        propsal = self.back.clone().detach() .view(batch,-1)
        conf = conf.view(batch,-1)
        loss = (x_saliency - propsal) * (x_saliency - propsal) 
        loss = torch.mean(loss, -1)
        loss = loss.mean(0)
        return loss
 
    def get_loss_2(self):    
        return self.seed_loss(self.back, self.back, self.back, 0.6, 0.6)

    def get_loss_3(self):
        return self.seed_loss(self.x_2_saliency, self.x_2_saliency, self.x_2_saliency, 0.7, 0.7, 1,1) 
    
    def get_loss_4(self):
        return self.seed_loss(self.x_saliency, self.x_saliency, self.x_saliency, 0.5, 0.5, 1,1) 
    
    def get_loss_5(self):
        batch, channel, h, w= self.back.size()
        h_parmater = 1
        back = self.back.clone().detach()
        x_2_saliency = self.x_3_saliency.clone()
        back = self.normalize_atten_maps(back)
        x_2_saliency = self.normalize_atten_maps(x_2_saliency)
        fg = x_2_saliency * back
        bg = x_2_saliency * (1-back)
        fg_score = nn.AvgPool2d(h)(fg) * h_parmater
        bg_score = nn.AvgPool2d(h)(bg) * h_parmater
        score = torch.cat([fg_score, bg_score],1).squeeze(-1).squeeze(-1)
        label = torch.zeros((batch)).cuda().long()
        loss_func =  nn.CrossEntropyLoss().cuda()
        loss = loss_func(score, label)
        return loss
    
    
    def self_correlation(self, source, target): ## generate source confidence 
        #source_normal = self.normalize_atten_maps(source)
        target_cor = self.iou(source, target)
        confidence = target_cor 

        output = source * confidence
        #print(confidence)
        #output = output.mean(1).unsqueeze(1)
        #output = self.normalize_atten_maps(output)
        return output

    
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
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)    

def model(args, pretrained=True):
    # base and dilation can be modified according to your needs
    model = Model(args)
    model.apply(weight_init)  #权值初始化
    pretrained_dict = torch.load('vgg16.pth')## 获得键名，例如：features.0.weight
    model_dict = model.state_dict() ## 存储网络结构名字和对应的参数(字典)
    model_conv_name = []

    for i, (k, v) in enumerate(model_dict.items()):
        model_conv_name.append(k)
    for i, (k, v) in enumerate(pretrained_dict.items()):
        if k.split('.')[0] != 'features':
            break
        if np.shape(model_dict[model_conv_name[i]]) == np.shape(v):
            model_dict[model_conv_name[i]] = v 
    model.load_state_dict(model_dict)
    print("pretrained weight load complete..")
    return model