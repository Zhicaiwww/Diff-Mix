# original code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
# CMO
# Copyright (c) 2022-present NAVER Corp.
# MIT License

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision.models import resnet50

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from dataset import IMBALANCE_DATASET_NAME_MAPPING
from downstream_tasks.imb_utils.moco_loader import GaussianBlur
from downstream_tasks.imb_utils.randaugment import rand_augment_transform
from downstream_tasks.imb_utils.util import *
from downstream_tasks.losses import BalancedSoftmaxLoss, LDAMLoss
from downstream_tasks.mixup import CutMix

NUM_CLASS_MAPPING = {
    "flower": 102,
    "cub": 200,
}
THRESHOLD_MAPPING = {"flower": (30, 10), "cub": (20, 5)}
best_acc1 = 0


def parse_args():

    parser = argparse.ArgumentParser(description="PyTorch Cifar Training")
    parser.add_argument("--root", default="./data/", help="dataset setting")
    parser.add_argument(
        "--dataset",
        default="cifar100",
        help="dataset setting",
        choices=("cifar100", "Imagenet-LT", "iNat18", "cub", "flower"),
    )
    parser.add_argument("--model", metavar="model", default="resnet32")
    parser.add_argument(
        "--num_classes", default=100, type=int, help="number of classes "
    )
    parser.add_argument(
        "--imb_factor", default=0.01, type=float, help="imbalance factor"
    )
    parser.add_argument(
        "--rand_number", default=0, type=int, help="fix random number for data sampling"
    )

    parser.add_argument(
        "--syndata_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--syndata_p",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--loss_type",
        default="CE",
        type=str,
        help="loss type / method",
        choices=("CE", "LDAM", "BS"),
    )
    parser.add_argument(
        "--train_rule",
        default="None",
        type=str,
        help="data sampling strategy for train loader",
        choices=("None", "CBReweight", "DRW"),
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr_steps",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument("--cos", action="store_true", help="use cosine LR")
    parser.add_argument("--use_weighted_syn", action="store_true", help="use cosine LR")
    parser.add_argument(
        "-b", "--batch_size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
    parser.add_argument("--gamma", default=1, type=float, help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=2e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    # data augmentation setting
    parser.add_argument(
        "--data_aug",
        default="CMO",
        type=str,
        help="data augmentation type",
        choices=("vanilla", "CMO"),
    )
    parser.add_argument(
        "--mixup_prob", default=0.5, type=float, help="mixup probability"
    )
    parser.add_argument(
        "--start_data_aug", default=3, type=int, help="start epoch for aug"
    )
    parser.add_argument(
        "--end_data_aug", default=3, type=int, help="how many epochs to turn off aug"
    )
    parser.add_argument(
        "--weighted_alpha",
        default=1,
        type=float,
        help="weighted alpha for sampling probability (q(1,k))",
    )
    parser.add_argument(
        "--beta", default=1, type=float, help="hyperparam for beta distribution"
    )
    parser.add_argument("--use_randaug", action="store_true")
    parser.add_argument("--use_cutmix", action="store_true", help="use cutmix")

    # etc.
    parser.add_argument(
        "--exp_str",
        default="0",
        type=str,
        help="number to indicate which experiment it is",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-p",
        "--print_freq",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 100)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--root_log", type=str, default="log")
    parser.add_argument("--root_model", type=str, default="checkpoint")

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    args.store_name = "_".join(
        [
            "Z",
            args.dataset,
            args.model,
            args.loss_type,
            "lr",
            str(args.lr),
            args.train_rule,
            args.data_aug,
            str(args.imb_factor),
            str(args.rand_number),
            str(args.mixup_prob),
            os.path.basename(str(args.syndata_dir)),
            str(args.syndata_p),
            str(args.gamma),
            args.exp_str,
        ]
    )
    prepare_folders(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


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
    return pixel_values, labels


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.model))
    num_classes = NUM_CLASS_MAPPING[args.dataset]

    model = resnet50(
        pretrained=True
    )  # to use more models, see https://pytorch.org/vision/stable/models.html
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True  #

    # use_norm = True if args.loss_type == 'LDAM' else False
    # model = models.__dict__[args.model](num_classes=num_classes, use_norm=use_norm)

    # print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:0")
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.use_randaug:
        print("use randaug!!")
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
            rand_augment_transform("rand-n{}-m{}-mstd0.5".format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        transform_train = [
            transforms.Compose(augmentation_randncls),
            transforms.Compose(augmentation_sim),
        ]

        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    print(args)

    synthetic_dir = None if args.syndata_dir is None else args.syndata_dir

    train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](
        split="train",
        imbalance_factor=args.imb_factor,
        weighted_alpha=args.weighted_alpha,
        seed=args.rand_number,
        synthetic_dir=synthetic_dir,
        syndata_p=args.syndata_p,
        gamma=args.gamma,
        use_randaug=args.use_randaug,
        use_weighted_syn=args.use_weighted_syn,
        return_onehot=True if synthetic_dir is not None else False,
    )
    train_dataset.transform = transform_train

    val_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](
        split="val", return_onehot=False
    )
    val_dataset.transform = transform_val
    # train_dataset = IMBALANCECIFAR100(root=args.root, imb_factor=args.imb_factor,
    #                                   rand_number=args.rand_number, weighted_alpha=args.weighted_alpha, train=True, download=True,
    #                                   transform=transform_train, use_randaug=args.use_randaug)
    cls_num_list = train_dataset.get_cls_num_list()

    if args.use_cutmix:
        train_dataset = CutMix(
            train_dataset, num_class=train_dataset.num_classes, prob=0.1
        )
    print("cls num list:")
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
    )

    weighted_train_loader = None

    if args.data_aug == "CMO":
        weighted_sampler = train_dataset.get_weighted_sampler()
        weighted_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=weighted_sampler,
        )

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    # init log for training
    log_training = open(
        os.path.join(args.root_log, args.store_name, "log_train.csv"), "w"
    )
    log_testing = open(
        os.path.join(args.root_log, args.store_name, "log_test.csv"), "w"
    )
    with open(os.path.join(args.root_log, args.store_name, "args.txt"), "w") as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    start_time = time.time()
    print("Training started!")

    for epoch in range(args.start_epoch, args.epochs):

        if args.use_randaug:
            paco_adjust_learning_rate(optimizer, epoch, args)
        else:
            adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == "None":
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == "CBReweight":
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == "DRW":
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn("Sample rule is not listed")

        if args.loss_type == "CE":
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == "BS":
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(
                args.gpu
            )
        elif args.loss_type == "LDAM":
            criterion = LDAMLoss(
                cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights
            ).cuda(args.gpu)
        else:
            warnings.warn("Loss type is not listed")
            return

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
            log_training,
            tf_writer,
            weighted_train_loader,
        )

        # evaluate on validation set
        acc1 = validate(
            val_loader, model, criterion, epoch, args, log_testing, tf_writer
        )

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar("acc/test_top1_best", best_acc1, epoch)
        output_best = "Best Prec@1: %.3f\n" % (best_acc1)
        print(output_best)
        log_testing.write(output_best + "\n")
        log_testing.flush()

        # save_checkpoint(args, {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc1': best_acc1,
        # }, is_best, epoch + 1)

    end_time = time.time()

    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    log_testing.write(
        "It took {} to execute the program".format(hms_string(end_time - start_time))
        + "\n"
    )
    log_testing.flush()


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.0
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    log,
    tf_writer,
    weighted_train_loader=None,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.train()

    end = time.time()
    if args.data_aug == "CMO" and args.start_data_aug < epoch < (
        args.epochs - args.end_data_aug
    ):
        inverse_iter = iter(weighted_train_loader)

    for i, (input, target) in enumerate(train_loader):
        if args.data_aug == "CMO" and args.start_data_aug < epoch < (
            args.epochs - args.end_data_aug
        ):

            try:
                input2, target2 = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_train_loader)
                input2, target2 = next(inverse_iter)

            input2 = input2[: input.size()[0]]
            target2 = target2[: target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # Data augmentation
        r = np.random.rand(1)

        if (
            args.data_aug == "CMO"
            and args.start_data_aug < epoch < (args.epochs - args.end_data_aug)
            and r < args.mixup_prob
        ):
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            # compute output
            output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (
                1.0 - lam
            )

        else:
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]["lr"],
                )
            )
            print(output)
            log.write(output + "\n")
            log.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)
    tf_writer.add_scalar("acc/train_top5", top5.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(
    val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag="val"
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = "{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}".format(
            flag="Epoch" + "[" + str(epoch) + "]" + flag,
            top1=top1,
            top5=top5,
            loss=losses,
        )
        out_cls_acc = "%s Class Accuracy: %s" % (
            flag,
            (
                np.array2string(
                    cls_acc,
                    separator=",",
                    formatter={"float_kind": lambda x: "%.3f" % x},
                )
            ),
        )
        print(output)
        # print(out_cls_acc)

        if args.imb_factor == 0.01:

            thresh_1, thresh_2 = THRESHOLD_MAPPING[args.dataset]
            many_shot = train_cls_num_list > thresh_1
            medium_shot = (train_cls_num_list <= thresh_1) & (
                train_cls_num_list >= thresh_2
            )
            few_shot = train_cls_num_list < thresh_2
            output_mmf = "many avg {:.3f}, med avg{:.3f}, few avg{:.3f}".format(
                float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
                float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
                float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)),
            )
            print(output_mmf)

        if log is not None:
            log.write(output + "\n")
            log.write(out_cls_acc + "\n")
            if args.imb_factor == 0.01:
                log.write(output_mmf + "\n")
            log.flush()

        tf_writer.add_scalar("loss/test_" + flag, losses.avg, epoch)
        tf_writer.add_scalar("acc/test_" + flag + "_top1", top1.avg, epoch)
        tf_writer.add_scalar("acc/test_" + flag + "_top5", top5.avg, epoch)
        tf_writer.add_scalars(
            "acc/test_" + flag + "_cls_acc",
            {str(i): x for i, x in enumerate(cls_acc)},
            epoch,
        )

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def paco_adjust_learning_rate(optimizer, epoch, args):
    # experiments as PaCo (ICCV'21) setting.
    warmup_epochs = 10

    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    if epoch <= warmup_epochs:
        lr = args.lr / warmup_epochs * (epoch + 1)
    elif epoch > 360:
        lr = args.lr * 0.01
    elif epoch > 320:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
