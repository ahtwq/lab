import argparse
import os
import sys
import time
import torch
from torch import nn
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import utils
import models
import tabulate
from PIL import Image, ImageFile
import numpy as np
import random
np.set_printoptions(suppress=True)


## args
parser = argparse.ArgumentParser(description='clarity training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='input batch size (default: 10)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
print(args)

## python train.py --dir=weight --epochs=25 --batch_size=20 --lr=0.001 

## seed fix
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

print('Preparing directory {}'.format(args.dir))
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

## Dataset
print('Loading dataset')
def default_loader(img):
    return Image.open(img)
    
class Mydataset(Dataset):
    def __init__(self, img_root, txtfile, img_transform=None, label_transform=None, loader=default_loader):
        with open(txtfile, 'r') as f:
            lines = f.readlines()
        self.img_list = [os.path.join(img_root, i.split()[0]+'.png') for i in lines]
        self.label_list = [np.float32(i.split()[1]) for i in lines]
                
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

                     
train_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomRotation(18, resample=Image.BICUBIC),
                        transforms.ColorJitter(0.15,0.15,0.15),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
val_transform =  transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])

img_root = '/Storage/DiamondData/bbg_png_1024_rotate10_new'
train_txt = '../DataTxt/dataset_uniform/train_1-10-0911_rotate*10.txt'
val_txt = '../DataTxt/dataset_uniform/val_1-10-0911_rotate*10.txt'

train_txt = val_txt
val_txt = '../DataTxt/dataset_uniform/test_0911_rotate*10.txt'

train_set = Mydataset(img_root=img_root, txtfile=train_txt, img_transform=train_transform)                       
test_set = Mydataset(img_root=img_root, txtfile=val_txt, img_transform=val_transform)

loaders = {
    'train': DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    ),
    'test': DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
}

## Model
print('Building model')
num_classes = 6
pre_model = models.EfficientNet
#pre_model = pre_model.from_pretrained('efficientnet-b1')
pre_model = pre_model.from_name('efficientnet-b0')
model = models.multiScale_Bx(pre_model, num_classes=1)
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('Using {} GPUs'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model.cuda()

## Loss func
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

## LR schedule
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5)

## Train model
print('Starting train model')
columns = ['ep', 'lr', 'tr_loss', 'tr_acc[%]', 'te_loss', 'te_acc[%]', 'time[m]']
class_names = ['VVS', 'VS1', 'VS2', 'SI1', 'SI2']
results = {}
best_acc_on_dev = 70.0
logger = utils.Logger(fpath=os.path.join(args.dir,'log.txt'))
logger.set_names(['epoch', 'Train Loss', 'Train Acc.', 'Valid Loss', 'Valid Acc.'])
writer = utils.wirter_cm(fpath=args.dir+'/train.csv')

for epoch in range(0, args.epochs):
	time_ep = time.time()
	scheduler.step()
	lr = optimizer.param_groups[0]['lr']
	train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, num_classes)
	test_res = utils.eval(loaders['test'], model, criterion, num_classes)

	logger.append([epoch, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy']])
	logger.plot()
	utils.savefig(os.path.join(args.dir, 'accuracy.eps'))
	
	infolist=['epoch{}'.format(epoch), 'train acc', train_res['accuracy'], 'test acc', test_res['accuracy']]
	cms = [train_res['conf_matrix'], test_res['conf_matrix']]
	writer.writer_in(cms, infolist,class_names=class_names)
	results['epoch'+str(epoch)] = [round(train_res['accuracy'],6), round(test_res['accuracy'],6)]
	
    ## Save model
	if test_res['accuracy'] >= best_acc_on_dev -2:
		best_acc_on_dev = test_res['accuracy']
		utils.save_checkpoint(args.dir, epoch, state_dict=model.state_dict())

	time_ep = (time.time() - time_ep) / 60
	values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]
	table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.6f')
	if epoch % 10 == 0:
		table = table.split('\n')
		table = '\n'.join([table[1]] + table)
	else:
		table = table.split('\n')[2]
	print(table)

print()
print('Train over. The top3 by test accuracy: ')
top3 = sorted(results.items(), key=lambda item:item[1][1], reverse=True)[0:3]
print(top3)
orders = [i[0][-1] for i in top3 if i[1][-1] > 0.0]

for i,item in enumerate(orders):
	print('top'+str(i))
	print('*'*30)
	dir = os.path.join(args.dir, 'checkpoint-{}.pt'.format(i))
	output_res = os.path.join(args.dir, 'test-{}.txt'.format(i))
	utils.predict(model, dir, output_res, num_classes)

