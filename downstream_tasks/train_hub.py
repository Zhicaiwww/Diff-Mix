import argparse
import os
import random
import shutil
import sys
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights, resnet18, resnet50, vit_b_16
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from dataset import DATASET_NAME_MAPPING
from downstream_tasks.losses import LabelSmoothingLoss
from downstream_tasks.mixup import CutMix, mixup_data
from utils.misc import checked_has_run
from utils.network import freeze_model

#######################
##### 1 - Setting #####
#######################


##### args setting
def formate_note(args):

    args.use_warmup = True
    note = f"{args.note}"
    if args.syndata_dir is not None:
        note = note + f"_{os.path.basename(args.syndata_dir[0])}"
    if args.use_cutmix:
        note = note + "_cutmix"
    if args.use_mixup:
        note = note + "_mixup"
    return note


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="cub", help="dataset name")
parser.add_argument(
    "--syndata_dir",
    type=str,
    nargs="+",
    help="key for indexing synthetic data",
)
parser.add_argument(
    "--syndata_p", default=0.1, type=float, help="synthetic probability"
)
parser.add_argument(
    "-m",
    "--model",
    default="resnet50",
    choices=["resnet50", "vit_b_16"],
    help="model name",
)
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch_size")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--use_cutmix", default=False, action="store_true")
parser.add_argument("--use_mixup", default=False, action="store_true")
parser.add_argument("--criterion", default="ls", type=str)
parser.add_argument("-g", "--gpu", default="1", type=int)
parser.add_argument("-w", "--num_workers", default=12, help="num_workers of dataloader")
parser.add_argument("-s", "--seed", default=2020, help="random seed")
parser.add_argument(
    "-n",
    "--note",
    default="",
    help="exp note, append after exp folder, fgvc(_r50) for example",
)
parser.add_argument(
    "-p",
    "--group_note",
    default="debug",
)
parser.add_argument(
    "-a",
    "--amp",
    default=0,
    help="0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp",
)
parser.add_argument(
    "-rs",
    "--resize",
    default=512,
    type=int,
)
parser.add_argument(
    "--res_mode",
    default="224",
    type=str,
)
parser.add_argument(
    "-cs",
    "--crop_size",
    type=int,
    default=448,
)
parser.add_argument(
    "--examples_per_class",
    type=int,
    default=-1,
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.0,
    help="label smoothing factor for synthetic data",
)
parser.add_argument(
    "-mp",
    "--mixup_probability",
    type=float,
    default=0.5,
)

parser.add_argument(
    "-ne",
    "--nepoch",
    type=int,
    default=448,
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
)

parser.add_argument(
    "-fs",
    "--finetune_strategy",
    type=str,
    default=None,
)
args = parser.parse_args()


##### exp setting


if args.optimizer == "sgd":
    base_lr = 0.02
elif args.optimizer == "adamw":
    base_lr = 1e-3
else:
    raise ValueError("optimizer not supported")

if args.res_mode == "28":
    args.resize = 32
    args.crop_size = 28
    args.batch_size = 2048
elif args.res_mode == "224":
    args.resize = 256
    args.crop_size = 224
    if args.model == "resnet50":
        args.batch_size = 256
    elif args.model == "vit_b_16":
        args.batch_size = 128
    else:
        raise ValueError("model not supported")
elif args.res_mode == "384":
    args.resize = 440
    args.crop_size = 384
    if args.model == "resnet50":
        args.batch_size = 128
    elif args.model == "vit_b_16":
        args.batch_size = 32
    else:
        raise ValueError("model not supported")
elif args.res_mode == "448":
    args.resize = 512
    args.crop_size = 448
    if args.model == "resnet50":
        args.batch_size = 64
    elif args.model == "vit_b_16":
        args.batch_size = 32
    else:
        raise ValueError("model not supported")
else:
    raise ValueError("res_mode not supported")


use_amp = int(args.amp)  # use amp to accelerate training

##### data settings
# data_dir = join('data', datasets_name)

lr_begin = args.lr
seed = int(args.seed)
datasets_name = args.dataset
num_workers = int(args.num_workers)
exp_dir = "outputs/result/{}/{}{}".format(
    args.group_note, datasets_name, formate_note(args)
)  # the folder to save model

# if checked_has_run(exp_dir, vars(args)):
#     exit()


##### CUDA device setting
torch.cuda.set_device(args.gpu)

##### Random seed setting
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


##### Dataloader setting
re_size = args.resize
crop_size = args.crop_size

if args.syndata_dir is not None:
    synthetic_dir = args.syndata_dir
else:
    synthetic_dir = None

return_onehot = True
gamma = args.gamma
synthetic_probability = args.syndata_p
examples_per_class = args.examples_per_class


def to_tensor(x):
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([to_tensor(example["label"]) for example in examples])
    dtype = torch.float32 if len(labels.size()) == 2 else torch.long
    labels.to(dtype=dtype)
    return {"pixel_values": pixel_values, "labels": labels}


train_set = DATASET_NAME_MAPPING[datasets_name](
    split="train",
    image_size=re_size,
    crop_size=crop_size,
    synthetic_dir=synthetic_dir,
    synthetic_probability=synthetic_probability,
    return_onehot=return_onehot,
    gamma=gamma,
    examples_per_class=examples_per_class,
)
test_set = DATASET_NAME_MAPPING[datasets_name](
    split="val", image_size=re_size, crop_size=crop_size, return_onehot=return_onehot
)

batch_size = min(args.batch_size, len(train_set))
nb_class = train_set.num_classes
if args.use_cutmix:
    train_set = CutMix(
        train_set, num_class=train_set.num_classes, prob=args.mixup_probability
    )
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
)
eval_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
)


MODEL_DICT = {
    "resnet50": resnet50,
    "resnet18": resnet18,
    "vit_b_16": vit_b_16,
}
##### Model settings
if args.model == "resnet18":
    net = resnet18(
        pretrained=True
    )  # to use more models, see https://pytorch.org/vision/stable/models.html
    net.fc = nn.Linear(
        net.fc.in_features, nb_class
    )  # set fc layer of model with exact class number of current dataset

elif args.model == "resnet50":
    net = resnet50(
        pretrained=True
    )  # to use more models, see https://pytorch.org/vision/stable/models.html
    net.fc = nn.Linear(net.fc.in_features, nb_class)
    # set fc layer of model with exact class number of current dataset

elif args.model == "vit_b_16":
    net = vit_b_16(
        weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
    )
    net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)

for param in net.parameters():
    param.requires_grad = True  # make parameters in model learnable

if args.finetune_strategy is not None and args.model == "resnet50":

    freeze_model(net, args.finetune_strategy)

##### optimizer setting
#
if args.criterion == "ce":
    criterion = nn.CrossEntropyLoss()
elif args.criterion == "ls":
    criterion = LabelSmoothingLoss(
        classes=nb_class, smoothing=0.1
    )  # label smoothing to improve performance
else:
    raise NotImplementedError

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr_begin,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr_begin,
    )
else:
    raise ValueError("optimizer not supported")

total_steps = args.nepoch * len(train_loader.dataset) // batch_size

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
import pytorch_warmup

warmup_scheduler = pytorch_warmup.LinearWarmup(
    optimizer, warmup_period=max(int(0.1 * total_steps), 1)
)


##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile(__file__, exp_dir + "/train_hub.py")
# save args to yaml
with open(os.path.join(exp_dir, "config.yaml"), "w+", encoding="utf-8") as file:
    yaml.dump(vars(args), file)

with open(os.path.join(exp_dir, "train_log.csv"), "w+", encoding="utf-8") as file:
    file.write("Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n")


##### Apex
if use_amp == 1:  # use nvidia apex.amp
    print("\n===== Using NVIDIA AMP =====")
    from apex import amp

    net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    with open(os.path.join(exp_dir, "train_log.csv"), "a+", encoding="utf-8") as file:
        file.write("===== Using NVIDIA AMP =====\n")
elif use_amp == 2:  # use torch.cuda.amp
    print("\n===== Using Torch AMP =====")
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    with open(os.path.join(exp_dir, "train_log.csv"), "a+", encoding="utf-8") as file:
        file.write("===== Using Torch AMP =====\n")


########################
##### 2 - Training #####
########################
net.cuda()
min_train_loss = float("inf")
max_eval_acc = 0

for epoch in range(args.nepoch):
    print("\n===== Epoch: {} =====".format(epoch))
    net.train()  # set model to train mode, enable Batch Normalization and Dropout
    lr_now = optimizer.param_groups[0]["lr"]
    train_loss = train_correct = train_total = idx = 0

    for batch_idx, (batch) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx
        optimizer.zero_grad()  # Sets the gradients to zero
        # inputs, targets = inputs.cuda(), targets.cuda()
        inputs = batch["pixel_values"].cuda()
        targets = batch["labels"].cuda()

        if args.use_mixup and np.random.rand() < args.mixup_probability:
            inputs, targets = mixup_data(
                inputs, targets, alpha=1.0, num_classes=nb_class
            )

        if inputs.shape[0] < batch_size:
            continue
        ##### amp setting
        if use_amp == 1:  # use nvidia apex.amp
            x = net(inputs)
            loss = criterion(x, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # use torch.cuda.amp
            with autocast():
                x = net(inputs)
                loss = criterion(x, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x = net(inputs)
            loss = criterion(x, targets)
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(x.data, 1)
        train_total += targets.size(0)
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, axis=1)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

        with warmup_scheduler.dampening():
            scheduler.step()
    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print(
        "Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f}% ({}/{})".format(
            lr_now, train_loss, train_acc, train_correct, train_total
        )
    )

    ##### Evaluating model with test data every epoch
    if epoch % 4 == 0:
        with torch.no_grad():
            net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
            eval_correct = eval_total = 0
            for _, (batch) in enumerate(tqdm(eval_loader, ncols=80)):

                inputs = batch["pixel_values"].cuda()
                targets = batch["labels"].cuda()

                x = net(inputs)
                _, predicted = torch.max(x.data, 1)
                eval_total += targets.size(0)
                if len(targets.shape) == 2:
                    targets = torch.argmax(targets, axis=1)
                eval_correct += predicted.eq(targets.data).cpu().sum()
            eval_acc = 100.0 * float(eval_correct) / eval_total
            print(
                "Test | Acc: {:.3f}% ({}/{})".format(eval_acc, eval_correct, eval_total)
            )

            ##### Logging
            with open(
                os.path.join(exp_dir, "train_log.csv"), "a+", encoding="utf-8"
            ) as file:
                file.write(
                    "{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n".format(
                        epoch, lr_now, train_loss, train_acc, eval_acc
                    )
                )
            ##### save model with highest acc
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                torch.save(
                    net.state_dict(),
                    os.path.join(exp_dir, "max_acc.pth"),
                    _use_new_zipfile_serialization=False,
                )


########################
##### 3 - Testing  #####
########################
print("\n\n===== TESTING =====")

with open(os.path.join(exp_dir, "train_log.csv"), "a") as file:
    file.write("===== TESTING =====\n")

##### load best model
net.load_state_dict(torch.load(join(exp_dir, "max_acc.pth")))
net.eval()  # set model to eval mode, disable Batch Normalization and Dropout

for data_set, testloader in zip(["train", "eval"], [train_loader, eval_loader]):
    test_loss = correct = total = 0
    with torch.no_grad():
        for _, (batch) in enumerate(tqdm(testloader, ncols=80)):
            inputs = batch["pixel_values"].cuda()
            targets = batch["labels"].cuda()
            inputs, targets = inputs.cuda(), targets.cuda()
            if len(targets.shape) == 2:
                targets = torch.argmax(targets, axis=1)
            x = net(inputs)
            _, predicted = torch.max(x.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100.0 * float(correct) / total
    print("Dataset {}\tACC:{:.2f}\n".format(data_set, test_acc))

    ##### logging
    with open(os.path.join(exp_dir, "train_log.csv"), "a+", encoding="utf-8") as file:
        file.write("Dataset {}\tACC:{:.2f}\n".format(data_set, test_acc))

    with open(
        os.path.join(exp_dir, "acc_{}_{:.2f}".format(data_set, test_acc)),
        "a+",
        encoding="utf-8",
    ) as file:
        # save accuracy as file name
        pass
