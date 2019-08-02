#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
import torch
from torch import nn
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import utils
import tabulate
import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import numpy as np
import random
np.set_printoptions(suppress=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True


## args
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='input batch size (default: 20), when you run the valid_complex function, the batch_size must be the default value 10')
parser.add_argument('--test_file', type=str, default='test', metavar='N', help='test file select (default: valid)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--output_res', type=str, default=None, help='output result')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
print(args)

## python predict.py --dir=weight/checkpoint-5.pt --batch_size=10 --test_file=test --output_res=test0911.txt
## hello
## seed fix
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

## dataset
print('Loading dataset')

def default_loader(img):
    return Image.open(img)

class Mydataset(Dataset):
    def __init__(self, img_root, txtfile, img_transform=None, label_transform=None, loader=default_loader):
        with open(txtfile, 'r') as f:
            lines = f.readlines()
        self.img_list = [os.path.join(img_root, i.split()[0]+'.png') for i in lines if os.path.exists(os.path.join(img_root, i.split()[0]+'.png'))]
        self.label_list = [np.float32(i.split()[1]) for i in lines if os.path.exists(os.path.join(img_root,i.split()[0]+'.png'))]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = self.loader(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)

                        
val_transform =  transforms.Compose([
                     transforms.Resize(512*2),
                     transforms.CenterCrop(448*2),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])


img_path = '/Storage/DiamondData/bbg_png_1024_rotate10_new'
#img_path = '/home/asus/tang/test/20180911-Bot-3repeat/1thRetakeRotation10'
if args.test_file == 'valid':
    test_txt = '../DataTxt/dataset_uniform/val_1-10-0911_rotate*10.txt'
else:
    test_txt = '../DataTxt/dataset_uniform/test_0911_rotate*10.txt'
    # test_txt = '../DataTxt/20180911-repeat3/same_rotate10.txt'
    # test_txt = '../DataTxt/20180911-repeat3/test0911_repeat10.txt'
    # test_txt = '../DataTxt/20180911-repeat3/version_test_rotate10.txt'
print('Test on ', os.path.join(img_path,test_txt))

test_set = Mydataset(img_root=img_path, txtfile=test_txt, label_transform=None, img_transform=val_transform)

loaders = {
    'test': DataLoader(
        test_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
}

# model
print('Building model')
num_classes = 6
pre_model = models.EfficientNet
pre_model = pre_model.from_name('efficientnet-b1')
# pre_model = pre_model.from_pretrained('efficientnet-b1')
model = models.multiScale_Bx(pre_model, num_classes=1)
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('Using {} GPUs'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model.cuda()

weight_best = torch.load(args.dir)['state_dict']
model.load_state_dict(weight_best)

# loss func
criterion = nn.L1Loss()
order = os.path.basename(args.dir)[-4]

test_res = utils.eval(loaders['test'], model, criterion, num_classes)
print('test_loss:{:.4f}, test_acc:{:4f}'.format(test_res['loss'], test_res['accuracy']))
print(test_res['conf_matrix'])

test_res_com = utils.eval_complex(loaders['test'], model, criterion, num_classes, args.output_res)
print('test_loss:{:.4f}, test_acc:{:4f}'.format(test_res['loss'], test_res['accuracy']))
print(test_res_com['conf_matrix'])

infolist=[os.path.basename(args.dir), 'test acc.', test_res['accuracy'], 'test acc. com.', test_res_com['accuracy']]
cms = [test_res['conf_matrix'], test_res_com['conf_matrix']]
utils.writer_cm(cms, infolist, csvname='{}/test{}.csv'.format(args.dir, order)), class_names=class_names)


