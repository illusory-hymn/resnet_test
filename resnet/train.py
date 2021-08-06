import os
import argparse
import torch
import torch.nn as nn 
from resnet import model
from DataLoader import ImageDataset
from torch.autograd import Variable
from utils.loss import Loss
from utils.accuracy import *
from utils.lr import *
import os
import random
import time
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
parser = argparse.ArgumentParser()
##  path
parser.add_argument('--root', type=str, default='/sdd/imagenet-1k')
parser.add_argument('--test_gt_path', type=str, default='val_gt.txt')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--test_txt_path', type=str, default='val_list.txt')

parser.add_argument('--save_path', type=str, default='logs')
parser.add_argument('--load_path', type=str, default='VGG.pth.tar')
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256) 
parser.add_argument('--tencrop', default=False) ## True / False
##  dataloader
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--nest', action='store_true')
##  train
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--pretrain', type=str, default='True')
parser.add_argument('--phase', type=str, default='train') ## train / test
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--power', type=float, default=0.9)
parser.add_argument('--momentum', type=float, default=0.9)
##  model
parser.add_argument('--model_name', type=str, default='VGG')
##  show
parser.add_argument('--show_step', type=int, default=94)
##  GPU'
parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lr = args.lr
def get_optimizer(model, args):
        lr =args.lr        
        weight_list = [] 
        bias_list = []
        last_weight_list = []
        last_bias_list =[]
        for name, value in model.named_parameters():
            if 'copy' in name : 
                continue
            if 'classifier' in name : 
                if 'weight' in name:
                    last_weight_list.append(value)
                elif 'bias' in name:
                    last_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)
        optmizer = torch.optim.SGD([{'params': weight_list,
                                     'lr': lr},
                                    {'params': bias_list,
                                     'lr': lr*2},
                                    {'params': last_weight_list,
                                     'lr': lr*10},
                                    {'params': last_bias_list,
                                     'lr': lr*20}], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        return optmizer


if __name__ == '__main__':
    '''print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)'''
    print(torch.cuda.get_device_name(0))
    if args.phase == 'train':
        ##  data
        MyData = ImageDataset(args)
        MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)
        ##  model
        model = model(args).cuda()
     #   model = nn.DataParallel(model, device_ids=[0,1])
        model.train()
        #model.load_state_dict(]torch.load('logs/VGG7.pth.tar'))
        ##  optimizer 
        optimizer = get_optimizer(model, args)
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nest)
        ##  Loss
        loss_func = nn.CrossEntropyLoss().cuda()
        epoch_loss = 0

        param_features = []
        param_classifiers = []
        param_branch = []

        print('Train begining!')
        for epoch in range(0, args.epochs):
            ##  accuracy
            cls_acc_1 = AverageMeter()
            cls_acc_2 = AverageMeter()
            cls_acc_3 = AverageMeter()
            loss_epoch_1 = AverageMeter()
            loss_epoch_2 = AverageMeter()
            loss_epoch_3 = AverageMeter()
            loss_epoch_4 = AverageMeter()
            loss_epoch_5 = AverageMeter()
            poly_lr_scheduler(optimizer, epoch, decay_epoch=10)
            poly_lr_scheduler(optimizer, epoch, decay_epoch=15)
           # print(float(model.w_linear.data.cpu()))
          #  torch.cuda.synchronize()
          #  start = time.time()

            for step, (path, imgs, label) in enumerate(MyDataLoader):
                
                imgs, label = Variable(imgs).cuda(), label.cuda()
                ##  backward
                optimizer.zero_grad()
                print(imgs)
                print(label)
                output1= model(imgs, label)
                label = label.long()
                pred = torch.max(output1, 1)[1]  
                loss_1 = loss_func(output1, label).cuda()
                loss_5 = loss_1#model.get_loss_1().cuda() 
                loss_6 = loss_1#model.get_loss_2().cuda() 
                loss_7 = loss_1#model.get_loss_3().cuda() 
                loss_8 = model.get_loss_4().cuda()
                loss =  loss_1   + loss_8
               # loss = loss_f
               # unc(output1, label).cuda() +  loss_func(output2, label).cuda() + loss_func(output3, label).cuda()
                loss.backward()
                optimizer.step() 
                ##  count_accuracy
                cur_batch = label.size(0)
                cur_cls_acc_1 = 100. * compute_cls_acc(output1, label) 
                cls_acc_1.updata(cur_cls_acc_1, cur_batch)
                loss_epoch_1.updata(loss_1.data, 1)
                loss_epoch_2.updata(loss_5.data, 1)
                loss_epoch_3.updata(loss_6.data, 1)
                loss_epoch_4.updata(loss_7.data, 1)
                loss_epoch_5.updata(loss_8.data, 1)
                if (step+1) % args.show_step == 0 :
                    print('Epoch:[{}/{}]\tstep:[{}/{}]   loss_epoch_1:{:.3f}\tloss_epoch_2:{:.3f}\tloss_epoch_3:{:.3f}\tloss_epoch_4:{:.3f}\tloss_epoch_5:{:.3f}\tepoch_acc_1:{:.2f}%'.format(
                            epoch+1, args.epochs, step+1, len(MyDataLoader), loss_epoch_1.avg,loss_epoch_2.avg, loss_epoch_3.avg, loss_epoch_4.avg,loss_epoch_5.avg,cls_acc_1.avg
                    ))
                    
                if step % 188 == 0:
                    iter = int(step // 188)
                    torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name + str(epoch)+'_'+ str(iter) +'.pth.tar'),_use_new_zipfile_serialization=False)
             #   torch.cuda.synchronize()
             #   end = time.time()
             #   print(end-start) 

            
    elif args.phase == 'test':
        ##  data
        MyData = ImageDataset(args)
        MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ##  model
        model = model(args, False).cuda()
        model.eval()
        model.load_state_dict(torch.load('logs/VGG99.pth.tar'))
        cls_acc_1 = AverageMeter()
        cls_acc_2 = AverageMeter()
        cls_acc_3 = AverageMeter()
        cls_acc_4 = AverageMeter()

        for step, (img, labels, bbox, path) in enumerate(MyDataLoader):
            with torch.no_grad():
                img, labels = Variable(img).cuda(), labels.cuda()
                output1 = model(img)
                #label =  torch.max(labels, 1)[1]
                ##  count_accuracy
                cur_batch =  labels.size(0)
                cur_cls_acc_1 = 100. * compute_cls_acc(output1, labels) 
                cls_acc_1.updata(cur_cls_acc_1, cur_batch)
                cur_cls_acc_2 = 100. * compute_cls_acc(output1, labels) 
                cls_acc_2.updata(cur_cls_acc_2, cur_batch)
                cur_cls_acc_3 = 100. * compute_cls_acc(output1, labels) 
                cls_acc_3.updata(cur_cls_acc_3, cur_batch)
                cur_cls_acc_4 = 100. * compute_cls_acc(output1, labels) 
                cls_acc_4.updata(cur_cls_acc_4, cur_batch)
                if (step+1) % args.show_step == 0:
                    print('cls_acc_1 : {} \t cls_acc_2 : {}\t cls_acc_3 : {}\t cls_acc_4 : {}'.format(cls_acc_1.avg, cls_acc_2.avg, cls_acc_3.avg, cls_acc_4.avg))
        print('cls_acc_1 : {} \t cls_acc_2 : {}\t cls_acc_3 : {}\t cls_acc_4 : {}'.format(cls_acc_1.avg, cls_acc_2.avg, cls_acc_3.avg, cls_acc_4.avg))
        





