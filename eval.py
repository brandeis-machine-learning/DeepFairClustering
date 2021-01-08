# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def predict(data_loader, encoder, dfc):
    features = []
    labels = []
    encoder.eval()
    dfc.eval()

    with torch.no_grad():
        for idx, (img, label) in enumerate(data_loader[0]):
            img = img.cuda()
            feat = dfc(encoder(img)[0])
            features.append(feat.detach())
            labels.append(label)

        for idx, (img, label) in enumerate(data_loader[1]):
            img = img.cuda()
            feat = dfc(encoder(img)[0])
            features.append(feat.detach())
            labels.append(label)

    return torch.cat(features).max(1)[1], torch.cat(labels).long()


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size

    return reassignment, accuracy


def entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy


def balance(predicted, size_0, k=10):
    count = torch.zeros((k, 2))
    for i in range(size_0):
        count[predicted[i], 0] += 1
    for i in range(size_0, predicted.shape[0]):
        count[predicted[i], 1] += 1

    count[count == 0] = 1e-5

    balance_0 = torch.min(count[:, 0] / count[:, 1])
    balance_1 = torch.min(count[:, 1] / count[:, 0])

    en_0 = entropy(count[:, 0] / torch.sum(count[:, 0]))
    en_1 = entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_0, balance_1).numpy(), en_0.numpy(), en_1.numpy()
