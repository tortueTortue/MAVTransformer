"""

"""

import torchvision.models as models
from models.mav_t import MAViT
from models.ViT import ViT_C
from models.ViT_2 import ViT

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import save_model, load_model, load_checkpoint
from training.metrics.metrics import print_accuracy_per_class, print_accuracy, count_model_parameters
import copy
import time

if __name__ == '__main__':
    # TODO Add as config
    model_dir = "E:/Git/MAVTransformer/models/trained_models/"
    checkpoint_dir = "E:/Git\MAVTransformer/training/checkpoints/checkpoint_mavt.pt"
    batch_size = 15
    epochs = 36
    cifar10_data = Cifar10Dataset(batch_size=batch_size)
    classes = cifar10_data.classes

    """
    In the paper introducing ViT "AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE
    RECOGNITION AT SCALE", the images from ImageNet were divided into 16 by 16 patches.
    """
    # MAViT ViT first
    # vitFirst = MAViT(32, 4, len(classes), 6 * 8, 3, 1,
    #              8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
    #              emb_dropout = 0., is_vit_first=True, batch_size=batch_size)
    # print(f"Parameters {count_model_parameters(vitFirst, False)}")
    # save_model(train_model(epochs, vitFirst, "vitFirst", cifar10_data, batch_size, model_dir), "vitFirst", model_dir)
    # # vF = load_checkpoint(copy.deepcopy(vitFirst), checkpoint_dir)
    # vitFirst = load_model(model_dir + "vitFirst.pt")
    # print_accuracy_per_class(vitFirst, classes, batch_size, cifar10_data.test_loader)
    # print_accuracy(vitFirst, classes, batch_size, cifar10_data.test_loader)


    vitFirst_XL = MAViT(32, 4, len(classes), 6 * 8, 6, 1,
                    8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                    emb_dropout = 0., is_vit_first=True, batch_size=batch_size)
    # vitFirst_XL = load_model(model_dir + "vitFirst_XL.pt")
    vitFirst_XL.train()
    print(f"Parameters {count_model_parameters(vitFirst_XL, False)}")
    start_time = time.time()
    save_model(train_model(epochs, vitFirst_XL, "vitFirst_XL", cifar10_data, batch_size, model_dir), "vitFirst_XL", model_dir)
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(vitFirst_XL, classes, batch_size, cifar10_data.test_loader)
    print_accuracy(vitFirst_XL, classes, batch_size, cifar10_data.test_loader)

    # vit Only 100 epochs
    # vit = ViT(32, 2, len(classes), 6 * 8, 12, 1, 8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.)
    # print(f"Parameters {count_model_parameters(vit, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, vit, "vitOnly", cifar10_data, batch_size, model_dir), "vitOnly", model_dir)
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(vit, classes, batch_size, cifar10_data.test_loader)
    # print_accuracy(vit, classes, batch_size, cifar10_data.test_loader)

    # MAT-XL LAT first
    # latFirst_XL = MAViT(32, 4, len(classes), 6 * 8, 6, 1,
    #                     8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
    #                     emb_dropout = 0., is_vit_first=False, batch_size=batch_size)
    # print(f"Parameters {count_model_parameters(latFirst_XL, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, latFirst_XL, "latFirst_XL", cifar10_data, batch_size, model_dir), "latFirst_XL", model_dir)
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(latFirst_XL, classes, batch_size, cifar10_data.test_loader)
    # print_accuracy(latFirst_XL, classes, batch_size, cifar10_data.test_loader)

    # ResNet34 100 epochs
    # resNet34 = models.resnet34(pretrained=False)
    # print(f"Parameters {count_model_parameters(resNet34, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, resNet34, "resNet34", cifar10_data, batch_size, model_dir), "resNet34", model_dir)
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(resNet34, classes, batch_size, cifar10_data.test_loader)
    # print_accuracy(resNet34, classes, batch_size, cifar10_data.test_loader)

    # MAViT LAT first
    # latFirst = MAViT(32, 4, len(classes), 8 * 8, 3, 1,
    #              8 * 8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
    #              emb_dropout = 0., is_vit_first=False)
    # print(f"Parameters {count_model_parameters(latFirst, False)}")
    # save_model(train_model(epochs, latFirst, "latFirst", cifar10_data, batch_size), "latFirst", model_dir)
    # print_accuracy_per_class(latFirst, classes, batch_size)

