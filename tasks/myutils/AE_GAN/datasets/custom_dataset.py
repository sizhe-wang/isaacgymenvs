import os
import torch
from torch.utils.data.dataset import Dataset

# root = self.params['data_path'],
#                              split = "train",
#                              transform=transform,
#                              download=False

class Custom_train_dataset(Dataset):
    def __init__(self, root=None, split='train', transform=None, download=False):
        file_path = os.path.join(root, 'image_tensors.pt')
        # self.data = torch.load(file_path)[:50040]
        self.data = torch.load(file_path)[:5004]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        target = img
        return img, target

    def __len__(self):
        return self.data.shape[0]


class Custom_val_dataset(Dataset):
    def __init__(self, root=None, split='train', transform=None, download=False):
        file_path = os.path.join(root, 'image_tensors.pt')
        self.data = torch.load(file_path)[50040:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        target = img
        # stuff
        return img, target

    def __len__(self):
        return self.data.shape[0]
