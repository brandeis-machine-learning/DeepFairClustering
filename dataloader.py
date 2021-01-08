# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

kwargs = {"shuffle": True, "num_workers": 4, "pin_memory": True, "drop_last": True}


class digital(data.Dataset):
    # with size in 32x32
    def __init__(self, subset, transform=None):
        file_dir = "./data/{}.txt".format(subset)
        self.data_dir = open(file_dir).readlines()
        self.transform = transform

    def __getitem__(self, index):
        img_dir, label = self.data_dir[index].split()
        img = Image.open(img_dir)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(np.int64(label)).long()

        return img, label

    def __len__(self):
        return len(self.data_dir)


def get_digital(args, subset):
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.5,), (0.5,))
                                    ])

    data = digital(subset, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.bs,
        **kwargs
    )

    return data_loader


def mnist_usps(args):
    train_0 = get_digital(args, "train_mnist")
    train_1 = get_digital(args, "train_usps")
    train_data = [train_0, train_1]

    return train_data
