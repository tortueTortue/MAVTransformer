import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.utils import make_grid

dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

labels = dataset.classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'deer', 'frog', 'horse', 'ship', 'truck' ]

torch.manual_seed(43)
train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.85),
                                                    int(len(dataset) * 0.15)])

batch_size=128

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

class Cifar10Dataset:
    def __init__(self, batch_size=128):
        dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
        test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
        
        self.labels = dataset.classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck' ]


        torch.manual_seed(43)
        train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.85),
                                                            int(len(dataset) * 0.15)])
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

    def get_dataloaders(self, batch=10):
        return self.train_loader, self.val_loader, self.test_loader


def run():
    torch.multiprocessing.freeze_support()
    for images, _ in train_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        break

# if __name__ == '__main__':
#     run()