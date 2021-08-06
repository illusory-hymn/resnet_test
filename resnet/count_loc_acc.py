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
from DataLoader import ImageDataset
from utils.accuracy import *
from utils.lr import *
from skimage import measure
from utils.func import *
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--test_gt_path', type=str, default='val_gt.txt')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--test_txt_path', type=str, default='val_list.txt')
parser.add_argument('--save_path', type=str, default='logs')
parser.add_argument('--load_path', type=str, default='VGG.pth.tar')
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256) 

parser.add_argument('--threshold', type=float, default=0.6)
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--init_weights', type=bool, default=False)
parser.add_argument('--phase', type=str, default='test')  
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

def save_error_img(path, heatmap, bbox, gt_box):
    path = str(path)[2:-3]
    #print(path)
    path = path.replace("\\\\", "/")
    ori_img = cv2.imread(str(path))
    name = path.split('/')[-1]

    cv2.imwrite('error_ori_img/'+name, ori_img)

    ori_img = cv2.resize(ori_img, (224, 224))
    heatmap = np.uint8(255 * cam)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_add = cv2.addWeighted(ori_img.astype(np.uint8), 0.5, heatmap.astype(np.uint8), 0.5, 0)

    cv2.rectangle(img_add, (bbox[0], bbox[1]),
                (bbox[2] , bbox[3]), (0, 255, 0), 4)
 ## 左上角坐标和右下角坐标'''
    ##  gt_box
    cv2.rectangle(img_add, (gt_box[0], gt_box[1]),
                (gt_box[2] , gt_box[3]), (0, 0, 255), 4)
    
    cv2.imwrite('error_img/'+name, img_add)

def normalize_map(atten_map):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val)/(max_val - min_val + 1e-5)
    atten_norm = cv2.resize(atten_norm, dsize=(224,224))
    return atten_norm
## 删除文件夹
if os.path.exists('error_img/'):  # 如果文件存在
    # 删除文件，可使用以下两种方法。
    shutil.rmtree('error_img/')  
    os.makedirs('error_img/')
##  data
MyData = ImageDataset(args)
MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
##  model 
model = model(args, False).cuda()
model.eval()
model.load_state_dict(torch.load('logs/VGG1_133.pth.tar'))
loc_acc = AverageMeter()
truncate = True
for step, (img, labels, gt_boxes, path) in enumerate(MyDataLoader): ## batch_size=1
    with torch.no_grad():
        img, labels = Variable(img).cuda(), labels.cuda()
        output1 = model(img, torch.tensor([0]))
        pred = torch.max(output1, 1)[1]


        ##  生成CAM
        fmap = model.get_fused_cam() 
        fmap = model.x_saliency[0] 
        #fmap = model.x_2_saliency[0] 
        #fmap = model.x_4_atten[0]
        #fmap = model.proposal[0] 
        #fmap = model.back[0]
       # fmap = model.get_loss()[0].data.cpu()
        fmap = np.array(fmap.data.cpu())
        cam = normalize_map(fmap[0])

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

        if len(props) == 0:
            #print(highlight)
            bbox = [0, 0, 224, 224]
        else:
            temp = props[0]['bbox']
            bbox = [temp[1], temp[0], temp[3], temp[2]] ##左上和右下坐标
            ##  cv2 x轴是横轴
        gt_bbox_num = len(gt_boxes)
        ##计算Loc_acc
        max_iou = 0
        max_box_num = 0
        for i in range(gt_bbox_num):
            iou = IoU(bbox, gt_boxes[i])
            if iou > max_iou:
                max_iou = iou
                max_box_num = i
        gt_boxes = gt_boxes[max_box_num]
        #print(bbox, gt_boxes)
        
        #if pred != labels:
        #   iou = 0
        if max_iou >= 0.5:
            loc_acc.updata(1, 1)
        else:
            loc_acc.updata(0, 1)
            #save_error_img(path, CAMs, bbox, gt_boxes)
    if (step+1) % 100 == 0:
        print("step:{} \t loc_acc:{}".format(step, loc_acc.avg))
print("loc_acc:{}".format(loc_acc.avg))

