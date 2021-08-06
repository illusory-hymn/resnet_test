import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
from utils.func import *
from utils.vis import *
from utils.IoU import *
from utils.augment import *
import argparse
from resnet import model
from skimage import measure

parser = argparse.ArgumentParser(description='Parameters for PSOL evaluation')
parser.add_argument('--loc-model', metavar='locarg', type=str, default='resnet50',dest='locmodel')
parser.add_argument('--cls-model', metavar='clsarg', type=str, default='vgg16',dest='clsmodel')
parser.add_argument('--input_size',default=256,dest='input_size')
parser.add_argument('--crop_size',default=224,dest='crop_size')
parser.add_argument('--num_classes',default=1000)
parser.add_argument('--tencrop', default=True)
parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
parser.add_argument('--data',metavar='DIR',default='./',help='path to imagenet dataset')
parser.add_argument('--threshold', type=float, default=0.2)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def normalize_map(atten_map,w,h):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val)/(max_val - min_val)
    atten_norm = cv2.resize(atten_norm, dsize=(w,h))
    return atten_norm
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data  
#os.environ['OMP_NUM_THREADS'] = "4"
#os.environ['MKL_NUM_THREADS'] = "4"
cudnn.benchmark = True
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose([
        transforms.Resize((args.crop_size,args.crop_size)),
     #   transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
cls_transform = transforms.Compose([
        transforms.Resize((args.crop_size,args.crop_size)),
     #   transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.TenCrop(args.crop_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])
model = model(args)
model.load_state_dict(torch.load('logs/VGG0_15.pth.tar'))
#print(model)
model = model.to(0)
model.eval()
cls_model = model
clsname = 'vgg'
root = args.data
val_imagedir = os.path.join(root, 'test')

anno_root = os.path.join(root,'bbox')
val_annodir = os.path.join(root, 'val_gt.txt')
val_list_path = os.path.join(root, 'val_list.txt')

classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()
#print(classes[0])


class_to_idx = {classes[i]:i for i in range(len(classes))}

result = {}

accs = []
accs_top5 = []
loc_accs = []
cls_accs = []
final_cls = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
bbox_f = open(val_annodir, 'r')
bbox_list = []
for line in bbox_f:
    part_1, part_2 = line.strip('\n').split(';')
    _, w, h, _ = part_1.split(' ')
    part_2 = part_2[1:]
    bbox = part_2.split(' ')
    bbox = np.array(bbox, dtype=np.float32)
    box_num = len(bbox) // 4
    w, h = np.float32(w),np.float32(h)
    for i in range(box_num):
        bbox[4*i], bbox[4*i+1], bbox[4*i+2], bbox[4*i+3] = bbox[4*i]/w, bbox[4*i+1]/h, bbox[4*i+2]/w , bbox[4*i+3]/h
    bbox_list.append(bbox)  ## gt
cur_num = 0
bbox_f.close()

files = [[] for i in range(1000)] ##[类别][路径]

with open(val_list_path, 'r') as f:
    for line in f:
        test_img_path, img_class =  line.strip("\n").split(';')
        files[int(img_class)].append(test_img_path)

for k in range(1000):
    cls = classes[k]

    total = 0
    IoUSet = []
    IoUSetTop5 = []
    LocSet = []
    ClsSet = []


    #files = os.listdir(os.path.join(val_imagedir, cls))
    #files.sort()

    for (i, name) in enumerate(files[k]):
        # raw_img = cv2.imread(os.path.join(imagedir, cls, name))

        gt_boxes = bbox_list[cur_num]
        cur_num += 1
        if len(gt_boxes)==0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, name)).convert('RGB')
        w, h = args.crop_size, args.crop_size

        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            reg_outputs = model(img)
            #bbox = to_data(reg_outputs)
            #bbox = torch.squeeze(bbox)
            #bbox = bbox.numpy()
           # cam = model.get_fused_cam5()[0].data.cpu()
            cam = model.x_2_saliency[0][0].data.cpu()
            cam = normalize_map(np.array(cam),w,h)
            
            ##  制作bbox
            highlight = np.zeros(cam.shape)
            highlight[cam > args.threshold] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(highlight, (w, h), interpolation=cv2.INTER_NEAREST)
            CAMs = copy.deepcopy(highlight_big)
            props = measure.regionprops(highlight_big.astype(int))

            if len(props) == 0:
                bbox = [0, 0, w, h]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]

            if TEN_CROP:
                img = ten_crop_aug(raw_img)
                img = img.to(0)
                vgg16_out = cls_model(img)
                vgg16_out = nn.Softmax()(vgg16_out)
                vgg16_out = torch.mean(vgg16_out,dim=0,keepdim=True)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            else:
                img = cls_transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                vgg16_out,_,_,_ = cls_model(img)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            vgg16_out = to_data(vgg16_out)
            vgg16_out = torch.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out
        ClsSet.append(out[0]==class_to_idx[cls])

        #handle resize and centercrop for gt_boxes

        gt_bbox_i = list(gt_boxes)
        raw_img_i = raw_img
      #  raw_img_i, gt_bbox_i = ResizedBBoxCrop((256,256))(raw_img, temp_list)
      #  raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i)
       # w, h = raw_img_i.size
        gt_bbox_i[0] = gt_bbox_i[0] * w
        gt_bbox_i[2] = gt_bbox_i[2] * w
        gt_bbox_i[1] = gt_bbox_i[1] * h
        gt_bbox_i[3] = gt_bbox_i[3] * h

        gt_boxes = gt_bbox_i

        w, h = raw_img_i.size

        bbox[0] = bbox[0]   #0和1左上角的点，原本的2和3是框的大小
        bbox[2] = bbox[2] 
        bbox[1] = bbox[1] 
        bbox[3] = bbox[3]  
        #print(gt_bbox_i, bbox)
        max_iou = -1
        iou = IoU(bbox, gt_boxes)
        if iou > max_iou:
            max_iou = iou

        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        #print(max_iou, name)
        result[os.path.join(cls, name)] = bbox   #max_iou
        IoUSet.append(max_iou)
        #cal top5 IoU
        max_iou = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
        IoUSetTop5.append(max_iou)
        #visualization code
    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    final_loc.extend(LocSet)
    cls_acc = np.sum(np.array(ClsSet))/len(ClsSet)
    final_cls.extend(ClsSet)
    print('{} cls-loc acc is {}, loc acc is {}, vgg16 cls acc is {}'.format(cls, cls_loc_acc, loc_acc, cls_acc))
    with open('inference_CorLoc.txt', 'a+') as corloc_f:
        corloc_f.write('{} {}\n'.format(cls, loc_acc))
    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    cls_accs.append(cls_acc)
    if (k+1) %100==0:
        print(k)


print(accs)
print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))

print('GT Loc acc {}'.format(np.mean(loc_accs)))
print('{} cls acc {}'.format(clsname, np.mean(cls_accs)))
with open('Corloc_result.txt', 'w') as f:
    for k in sorted(result.keys()):
        f.write('{} {}\n'.format(k, str(result[k])))
