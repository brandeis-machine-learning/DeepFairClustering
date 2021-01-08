# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import random
import numpy as np
import torch
from torch import nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv2d") != -1 or layer_name.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(layer.weight)
    elif layer_name.find("BatchNorm") != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_normal_(layer.weight)


def inv_lr_scheduler(optimizer, lr, iter, max_iter, gamma=10, power=0.75):
    learning_rate = lr * (1 + gamma * (float(iter) / float(max_iter))) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group["lr_mult"]
        i += 1

    return optimizer


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def aff(input):
    return torch.mm(input, torch.transpose(input, dim0=0, dim1=1))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
