import argparse
import os
import random
import shutil
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torchvision.models import resnet50
from tqdm import tqdm

from LabelSmoothing import LabelSmoothingLoss

# python train.py -d cub -b 64 -g 6 -w 12 -s 2020 -a 2 -n train_baseline

#######################
##### 1 - Setting #####
#######################

##### args setting
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='cub', help='dataset dir')
parser.add_argument('-b', '--batch_size', default=64, help='batch_size')
parser.add_argument(
    '-g', '--gpu', default='0', help='example: 0 or 1, to use different gpu'
)
parser.add_argument('-w', '--num_workers', default=12, help='num_workers of dataloader')
parser.add_argument('-s', '--seed', default=2020, help='random seed')
parser.add_argument(
    '-n',
    '--note',
    default='',
    help='exp note, append after exp folder, fgvc(_r50) for example',
)
parser.add_argument(
    '-a',
    '--amp',
    default=0,
    help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp',
)
args = parser.parse_args()


##### exp setting
seed = int(args.seed)
datasets_dir = args.dir
nb_epoch = 128  # 128 as default to suit scheduler
batch_size = int(args.batch_size)
num_workers = int(args.num_workers)
lr_begin = (batch_size / 256) * 0.1  # learning rate at begining
use_amp = int(args.amp)  # use amp to accelerate training


##### data settings
# data_dir = join('data', datasets_dir)
data_dir = '/data/zhicai/datasets/fgvc_datasets/CUB_200_2011/split'
data_sets = ['train', 'test']
nb_class = len(
    os.listdir(join(data_dir, data_sets[0]))
)  # get number of class via img folders automatically
exp_dir = 'result/{}{}'.format(datasets_dir, args.note)  # the folder to save model


##### CUDA device setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


##### Dataloader setting
re_size = 512
crop_size = 448

train_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.RandomCrop(crop_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

train_set = ImageFolder(root=join(data_dir, data_sets[0]), transform=train_transform)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)


##### Model settings
net = resnet50(
    pretrained=True
)  # to use more models, see https://pytorch.org/vision/stable/models.html
net.fc = nn.Linear(
    net.fc.in_features, nb_class
)  # set fc layer of model with exact class number of current dataset

for param in net.parameters():
    param.requires_grad = True  # make parameters in model learnable


##### optimizer setting
LSLoss = LabelSmoothingLoss(
    classes=nb_class, smoothing=0.1
)  # label smoothing to improve performance
optimizer = torch.optim.SGD(
    net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)


##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile('train.py', exp_dir + '/train.py')
shutil.copyfile('LabelSmoothing.py', exp_dir + '/LabelSmoothing.py')

with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n')


##### Apex
if use_amp == 1:  # use nvidia apex.amp
    print('\n===== Using NVIDIA AMP =====')
    from apex import amp

    net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using NVIDIA AMP =====\n')
elif use_amp == 2:  # use torch.cuda.amp
    print('\n===== Using Torch AMP =====')
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using Torch AMP =====\n')


########################
##### 2 - Training #####
########################
net.cuda()
min_train_loss = float('inf')
max_eval_acc = 0

for epoch in range(nb_epoch):
    print('\n===== Epoch: {} ====='.format(epoch))
    net.train()  # set model to train mode, enable Batch Normalization and Dropout
    lr_now = optimizer.param_groups[0]['lr']
    train_loss = train_correct = train_total = idx = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx

        if inputs.shape[0] < batch_size:
            continue

        optimizer.zero_grad()  # Sets the gradients to zero
        inputs, targets = inputs.cuda(), targets.cuda()

        ##### amp setting
        if use_amp == 1:  # use nvidia apex.amp
            x = net(inputs)
            loss = LSLoss(x, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # use torch.cuda.amp
            with autocast():
                x = net(inputs)
                loss = LSLoss(x, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x = net(inputs)
            loss = LSLoss(x, targets)
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(x.data, 1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

    scheduler.step()

    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print(
        'Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f}% ({}/{})'.format(
            lr_now, train_loss, train_acc, train_correct, train_total
        )
    )

    ##### Evaluating model with test data every epoch
    with torch.no_grad():
        net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
        eval_set = ImageFolder(
            root=join(data_dir, data_sets[-1]), transform=test_transform
        )
        eval_loader = DataLoader(
            eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        eval_correct = eval_total = 0
        for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = net(inputs)
            _, predicted = torch.max(x.data, 1)
            eval_total += targets.size(0)
            eval_correct += predicted.eq(targets.data).cpu().sum()
        eval_acc = 100.0 * float(eval_correct) / eval_total
        print(
            '{} | Acc: {:.3f}% ({}/{})'.format(
                data_sets[-1], eval_acc, eval_correct, eval_total
            )
        )

        ##### Logging
        with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
            file.write(
                '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n'.format(
                    epoch, lr_now, train_loss, train_acc, eval_acc
                )
            )

        ##### save model with highest acc
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            torch.save(
                net.state_dict(),
                os.path.join(exp_dir, 'max_acc.pth'),
                _use_new_zipfile_serialization=False,
            )


########################
##### 3 - Testing  #####
########################
print('\n\n===== TESTING =====')

with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
    file.write('===== TESTING =====\n')

##### load best model
net.load_state_dict(torch.load(join(exp_dir, 'max_acc.pth')))
net.eval()  # set model to eval mode, disable Batch Normalization and Dropout

for data_set in data_sets:
    testset = ImageFolder(
        root=os.path.join(data_dir, data_set), transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loss = correct = total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(testloader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = net(inputs)
            _, predicted = torch.max(x.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100.0 * float(correct) / total
    print('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

    ##### logging
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

    with open(
        os.path.join(exp_dir, 'acc_{}_{:.2f}'.format(data_set, test_acc)), 'a+'
    ) as file:
        # save accuracy as file name
        pass
