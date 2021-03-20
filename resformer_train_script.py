# https://www.kaggle.com/puneet6060/intel-image-classification
from __future__ import print_function, division
import os
import torchvision.models as models
import torch
from torch.nn import CrossEntropyLoss, Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, Softmax, Module
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from utils.utils import get_default_device, batches_to_device, to_device
import torch.utils.data as data
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


TRAIN_FOLDER_PATH = "E:/Image Datasets/Intel Scenes/archive/seg_train/seg_train"
TEST_FOLDER_PATH = "E:/Image Datasets/Intel Scenes/archive/seg_test/seg_test"

def load_intel_scene_data(batch=30, root_path="E:/Image Datasets/Intel Scenes/"):
    transform = transforms.Compose([transforms.Resize((150, 150)),
                                    transforms.ToTensor()])

    dataset = ImageFolder(root=TRAIN_FOLDER_PATH, transform=transform)
    
    torch.manual_seed(55)
    train_dataset, val_dataset = random_split(dataset, [11928, len(dataset) - 11928])

    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,  num_workers=4)

    test_data = ImageFolder(root=TEST_FOLDER_PATH, transform=transform)
    test_data_loader  = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=4)

    return train_data_loader, val_data_loader, test_data_loader


"""
 Architecture a ResNet18 to Extract a representation then sent to a patch tokenizer and 
 a Transformer encoder of 3 layers and an MLP.
 """

class Flatten(Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

class Flatten2(Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(x.shape[0], x.shape[1] * x.shape[2])

# Instantiate ResNet
resnet18 = models.resnet18(pretrained=True)

# Remove last layers
resnet18_last_layer_removed = list(resnet18.children())[:-2]
resnet18_last_layer_removed = Sequential(*resnet18_last_layer_removed)

# Check reprensentation dimension
data = torch.ones(1, 3 , 150, 150, requires_grad=True)
resnet18_last_layer_removed.eval()
output = resnet18_last_layer_removed(data)

# Instantiate Encoder
encoder_block = TransformerEncoderLayer(25, 5, dim_feedforward=2048, dropout=0.1, activation='relu')
encoder = TransformerEncoder(encoder_block, 3, norm=None)
encoder.eval()

data = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3], requires_grad=False)
data = torch.reshape(data, (512, 25)).unsqueeze(0)
# print(f"data new shape {data.shape}")
output = encoder(data)
# print(output)
# print(output.shape)

# Instantiate Patches tokeniser
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

# Instantiate or linear
# flat = Flatten()
lin = Linear(25*512, 10, True)
# output = lin(output)
# print(output)
# print(output.shape)

# Softmax
# soft = Softmax(dim=10)
# output = soft(output)
# print(output)
# print(output.shape)

# Build model
resformer = Sequential(resnet18_last_layer_removed,
                        Flatten(),
                        encoder,
                        Flatten2(),
                        lin)

data = torch.ones(1, 3, 150, 150, requires_grad=True)

# print(resformer(data))

"""
    T R A I N N I N G
"""
train_loader, val_loader, test_loader = load_intel_scene_data()

loss = CrossEntropyLoss()

resformer.train()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
    images, labels = batch 
    images, labels = images.cuda(), labels.cuda()
    out = model.forward(images)
    cross_entropy = CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: Module, val_set: DataLoader):
    outputs = [validation_step(model, batch) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def train(epochs_no: int, model: Module, train_set: DataLoader, val_set: DataLoader):
    history = []
    
    # TODO Read about optimizer optimizer = opt_func(model.parameters(), lr)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs_no):
        """  Training Phase """ 
        for batch in train_set:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()
            out = model.forward(inputs)
            curr_loss = loss(out, labels)
            curr_loss.backward()
            optimizer.step()
            


        """ Validation phase """
        result = evaluate(model, val_set)
        print(result)
        history.append(result)
        if epoch % 10 == 0 :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': curr_loss,
                }, 'E:/dxlat/Training stats/model.pt')
    return history

def train_model(epochs_no, model_to_train, name: str):
    device = get_default_device()

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    model = to_device(model_to_train, device)

    train(epochs_no, model, train_loader, val_loader)

    torch.save(model, f'E:/dxlat/Training stats/{name}.pt')

if __name__ == "__main__":
    device = get_default_device()
    #train_model(100, resformer, 'resformer')
    res50 = models.resnet50(pretrained=False)
    res50.train()
    # train_model(100, res50, 'resnet50')
    # checkpoint = torch.load('E:/dxlat/Training stats/resnet50.pt')

    # import copy
    # model = copy.deepcopy(resformer)
    # # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    # model = to_device(model, device)

    # model = torch.load('E:/dxlat/Training stats/resformer.pt')
    # model.eval()

    # print("Resformer")
    # class_correct = list(0. for i in range(6))
    # class_total = list(0. for i in range(6))
    # with torch.no_grad():
    #     for batch in test_loader:
    #         i, l = batch
    #         i, l = i.cuda(), l.cuda()
    #         out = model(i)
    #         _, predicted = torch.max(out, 1)
    #         c = (predicted == l).squeeze()
    #         for i in range(30):
    #             label = l[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    # classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    # for i in range(6):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))

    print("resnet 50")
    checkpoint = torch.load('E:/dxlat/Training stats/model.pt')
    import copy
    res50 = copy.deepcopy(res50)
    res50.load_state_dict(checkpoint['model_state_dict'])
    res50 = to_device(res50, device)
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    with torch.no_grad():
        for batch in test_loader:
            i, l = batch
            i, l = i.cuda(), l.cuda()
            out = res50(i)
            _, predicted = torch.max(out, 1)
            c = (predicted == l).squeeze()
            for i in range(30):
                label = l[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    for i in range(6):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

#TODO Clean code, save models, count parameters in read me, give no of epochs, trynet Resformer 34 resnet 18 et resent 34