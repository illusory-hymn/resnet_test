import torch.nn as nn
import torch 
from torch.autograd import Variable
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
from resnet import model
import argparse
from skimage import measure
from utils.func import *
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--thr', type=float, default=0.9)
parser.add_argument('--drop_prob', type=float, default=0.25)
parser.add_argument('--phase', type=str, default='test')
parser.add_argument('--push_thr', type=float, default=0.7)
parser.add_argument('--pull_thr', type=float, default=0.4)
parser.add_argument('--threshold', type=float, default=0.3)
args = parser.parse_args()

##  读入待预测的图片
def img_processing(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ##  Normalize参数是因为ImageNet数据集，我们使用的权重是这个数据集训练得到的
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()
    return img

def normalize_map(atten_map):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val)/(max_val - min_val + 1e-10)
    atten_norm = cv2.resize(atten_norm, dsize=(224,224))
    
    return atten_norm

##  模型载入

model = model(args, pretrained=False).cuda()
model.eval()


model.load_state_dict(torch.load('logs/VGG1_120.pth.tar')) 

##  图片读取
img_path = 'ILSVRC2012_val_00017699.jpg'
if os.path.exists(img_path):#
    pass
else:
    img_path = img_path[:-3] + 'JPEG'
ori_img = cv2.imread(img_path) 
ori_img = cv2.resize(ori_img, (224, 224))

img = img_processing(img_path)
label = [0]
output1= model(img, label)
label = torch.max(output1, 1)[1]

print(label)
##  生成CAM
cam = model.get_fused_cam()[0].data.cpu()
#cam = model.x_4_norm[0][0].data.cpu() 
#cam = model.x_erase_sum[0][0].data.cpu() 
cam = model.x_saliency[0][0].data.cpu() 
#cam = model.x_2_saliency[0][0].data.cpu() 
#cam = model.x_sum[0][0].data.cpu() 
#cam = model.back[0][0].data.cpu()
#cam = model.proposal[0][0].data.cpu() 
#cam = model.back_erase[0][0].data.cpu()
#cam = model.back_loss[0][0].data.cpu()
#cam = model.x_saliency[0][0].data.cpu()
val = model.get_loss_4()
print(val)
#cam = model.get_loss_1()[0][0].data.cpu()
#print(torch.sum(cam.view(-1),0))
#cam = model.x_atten[0][0].data.cpu()
cam = normalize_map(np.array(cam))
#cam[cam>0.8] = 1
#cam[cam<1] = 0
##  生成热力图
heatmap = np.uint8(255 * cam)
heatmap = heatmap.astype(np.uint8)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
img_add = cv2.addWeighted(ori_img.astype(np.uint8), 0.5, heatmap.astype(np.uint8), 0.5, 0)

##  制作bbox
highlight = np.zeros(cam.shape)
highlight[cam > args.threshold] = 1
# max component
all_labels = measure.label(highlight)
highlight = np.zeros(highlight.shape)
highlight[all_labels == count_max(all_labels.tolist())] = 1
highlight = np.round(highlight * 255)
highlight_big = cv2.resize(highlight, (224, 224), interpolation=cv2.INTER_NEAREST)
CAMs = copy.deepcopy(highlight_big)
props = measure.regionprops(highlight_big.astype(int))

'''
if len(props) == 0:
    #print(highlight)
    bbox = [0, 0, 224, 224]
else:
    temp = props[0]['bbox']
    bbox = [temp[1], temp[0], temp[3], temp[2]]
    print(bbox)'''


##  draw_bbox
'''cv2.rectangle(img_add, (bbox[0], bbox[1]),
                (bbox[2] , bbox[3]), (0, 255, 0), 4) ## 左上角坐标和右下角坐标'''
'''##  gt_box
gt_box = (46,35,192,210)
cv2.rectangle(img_add, (gt_box[0], gt_box[1]),
                (gt_box[2] , gt_box[3]), (0, 0, 255), 4)'''
#cv2.imshow('aa', img_add)
#cv2.imwrite('output/'+img_path, img_add)
#cv2.waitKey(0)
img_add = img_add[:,:,::-1]
plt.imshow(img_add)
plt.axis('off')
#plt.savefig('output/' + img_path)
plt.show()
