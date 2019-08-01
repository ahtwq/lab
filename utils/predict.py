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
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])
img_path = '/Storage/DiamondData/bbg_png_1024_rotate10_new'
test_txt = '../DataTxt/dataset_uniform/test_0911_rotate*10.txt'

def predict(model, dir, output_res=None, batch_size=10):
	test_set = Mydataset(img_root=img_path, txtfile=test_txt, label_transform=None, img_transform=val_transform)
	loaders = {
		'test': DataLoader(
			test_set,
			batch_size,
			shuffle=False,
			num_workers=4,
		)
	}
	# model
	weight_best = torch.load(dir)['state_dict']
	model.load_state_dict(weight_best)

	# loss func
	criterion = nn.L1Loss()
	order = os.path.basename(dir)[-4]

	test_res = utils.eval(loaders['test'], model, criterion, num_classes)
	print('test_loss:{:.4f}, test_acc:{:4f}'.format(test_res['loss'], test_res['accuracy']))
	print(test_res['conf_matrix'])

	test_res_com = utils.eval_complex(loaders['test'], model, criterion, num_classes, output_res)
	print('test_loss:{:.4f}, test_acc:{:4f}'.format(test_res['loss'], test_res['accuracy']))
	print(test_res_com['conf_matrix'])

	infolist=[os.path.basename(args.dir), 'test acc.', test_res['accuracy'], 'test acc. com.', test_res_com['accuracy']]
	cms = [test_res['conf_matrix'], test_res_com['conf_matrix']]
	utils.writer_cm(cms, infolist, csvname='{}/test-{}.csv'.format(args.dir, order), class_names=class_names)


