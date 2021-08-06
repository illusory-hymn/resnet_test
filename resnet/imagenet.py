import os
from shutil import copy

image_path = 'val'
val_path = 'val_list.txt'
classes = []
f = open(val_path, 'w')

image_dir = os.listdir(image_path)
for cla in image_dir:
    classes.append(cla)  ## 获取classes
class_num = len(classes)

for i in range(class_num):
    ##  逐一打开文件夹
    path = image_path + '/' + classes[i]
    cls_dir = os.listdir(path)
    for j in cls_dir:
        w_str = classes[i] + '/' + j + ';' + str(i) + '\n'
        f.write(w_str)
f.close()