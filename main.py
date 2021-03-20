"""
Trying out the Kaggle Intel Scene classification challenge using a Transformer Encoder/ Residual Network Hybrid.

Kaggle Intel Challenge
https://www.kaggle.com/puneet6060/intel-image-classification
"""

import torchvision.models as models
from 

from training_manager.training_manager import train_model
from dataset.intel_dataset import IntelScenesDataset
from utils.utils import save_model
from metrics.metrics import print_accuracy_per_class


if __name__ == '__main__':
    # TODO Add as config
    model_dir = ""
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    batch_size = 30
    epochs = 100
    intel_data = IntelScenesDataset()

    # ResNet18
    resNet18 = models.resnet34(pretrained=True)
    save_model(train_model(epochs, resNet18, "ResNet18", intel_data), "ResNet18", model_dir, batch_size)
    print_accuracy_per_class(resNet18, classes, batch_size)

    # ResNet34
    resNet34 = models.resnet34(pretrained=True)
    save_model(train_model(epochs, resNet34, "ResNet34", intel_data), "ResNet34", model_dir, batch_size)
    print_accuracy_per_class(resNet34, classes, batch_size)

    # ResNet50
    resNet50 = models.resnet50(pretrained=True)
    save_model(train_model(epochs, resNet50, "ResNet50", intel_data), "ResNet50", model_dir, batch_size)
    print_accuracy_per_class(resNet50, classes, batch_size)

    # Resformer-18
    resNet50 = models.resnet50(pretrained=True)
    save_model(train_model(epochs, resNet50, "Resformer-18", intel_data), "Resformer-18", model_dir, batch_size)
    print_accuracy_per_class(resNet50, classes, batch_size)

    # Resformer-34
    resNet50 = models.resnet50(pretrained=True)
    save_model(train_model(epochs, resNet50, "Resformer-34", intel_data), "Resformer-34", model_dir, batch_size)
    print_accuracy_per_class(resNet50, classes, batch_size)
