'''
Code from https://blog.csdn.net/Najlepszy/article/details/100540130
'''

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    # "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (M,N)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(1) == self.size
        x = x.log()
        true_dist = x.data.clone()  # 先深复制过来
        # print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        self.true_dist = true_dist
        print(x.shape, true_dist.shape)

        return self.criterion(x, Variable(true_dist, requires_grad=False))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            if target.dim() == 1:
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            else:
                true_dist = true_dist + target * self.confidence 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == "__main__":
    # Example of label smoothing.
    crit = LabelSmoothingLoss(classes=5, smoothing=0.1)
    # predict.shape 3 5
    predict = torch.FloatTensor(
        [[0, 0.2, 0.7, 0.1, 0], [0, 0.9, 0.2, 0.1, 0], [1, 0.2, 0.7, 0.1, 0]]
    )
    v = crit(Variable(predict), Variable(torch.LongTensor([2, 1, 0])))
    print(v)
    # Show the target distributions expected by the system.
    # plt.imshow(crit.true_dist)
