"""

"""

import torchvision.models as models
from models.mav_t import MAViT

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import save_model
from training.metrics.metrics import print_accuracy_per_class, count_model_parameters


if __name__ == '__main__':
    # TODO Add as config
    model_dir = ""
    batch_size = 5
    epochs = 100
    cifar10_data = Cifar10Dataset(batch_size=batch_size)
    classes = cifar10_data.classes

    """
    In the paper introducing ViT "AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE
    RECOGNITION AT SCALE", the images from ImageNet were divided into 16 by 16 patches.
    """
    # MAViT ViT first
    vitFirst = MAViT(32, 4, len(classes), 6 * 8, 3, 1,
                 8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., is_vit_first=True, batch_size=batch_size)
    print(f"Parameters {count_model_parameters(vitFirst, False)}")
    save_model(train_model(epochs, vitFirst, "vitFirst", cifar10_data, batch_size), "vitFirst", model_dir)
    print_accuracy_per_class(vitFirst, classes, batch_size)

    # MAViT LAT first
    latFirst = MAViT(32, 4, len(classes), 8 * 8, 3, 1,
                 8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., is_vit_first=False)
    print(f"Parameters {count_model_parameters(latFirst, False)}")
    save_model(train_model(epochs, latFirst, "latFirst", cifar10_data, batch_size), "latFirst", model_dir)
    print_accuracy_per_class(latFirst, classes, batch_size)

