# https://www.kaggle.com/puneet6060/intel-image-classification

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# TODO put in config file
TRAIN_FOLDER_PATH = "E:/Image Datasets/Intel Scenes/archive/seg_train/seg_train"
TEST_FOLDER_PATH = "E:/Image Datasets/Intel Scenes/archive/seg_test/seg_test"

class IntelScenesDataset(Dataset):
    """Intel Scenes dataset from Kaggle."""

    def __init__(self, csv_file, train_root_dir: str, test_root_dir: str, transform=None):
        """
        Args:
            train_root_dir (string): Directory with all the training images.
            test_root_dir (string): Directory with all the testing images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.test_root_dir = test_root_dir
        self.train_root_dir = train_root_dir
        self.transform = transform

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_dataloaders(self, batch=10):
        transform = transforms.Compose([transforms.ToTensor()]) if self.transform is None else self.transform

        dataset = ImageFolder(root=self.train_root_dir, transform=transform)
        
        torch.manual_seed(43)
        train_dataset, val_dataset = random_split(dataset, [11928, len(dataset) - 11928])

        train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=4)
        val_data_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,  num_workers=4)

        test_data = ImageFolder(root=self.test_root_dir, transform=transform)
        test_data_loader  = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=4)

        return train_data_loader, val_data_loader, test_data_loader
