"""
 Architecture a ResNet18 to Extract a representation then sent to a patch tokenizer and 
 a Transformer encoder of 3 layers and an MLP.
 """

import torchvision.models as models
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, Softmax, Flatten
from ..dataset.intel_dataset import load_intel_scene_data

# Instantiate ResNet
resnet18 = models.resnet18()

# Verify layers
print("ResNet before layer removal")
print(resnet18)

# Remove last layers
resnet18_last_layer_removed = list(resnet18.children())[:-2]
resnet18_last_layer_removed = Sequential(*resnet18_last_layer_removed)

# Verify layers
print("ResNet after layer removal")
print(resnet18_last_layer_removed)

# Check reprensentation dimension
data = torch.ones(1, 3 , 150, 150, requires_grad=True)
resnet18_last_layer_removed.eval()
output = resnet18_last_layer_removed(data)
print(output)
print(output.shape)


# Instantiate Encoder
encoder_block = TransformerEncoderLayer(25, 5, dim_feedforward=2048, dropout=0.1, activation='relu')
encoder = TransformerEncoder(encoder_block, 3, norm=None)
encoder.eval()

data = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3], requires_grad=False)
data = torch.reshape(data, (512, 25)).unsqueeze(0)
print(f"data new shape {data.shape}")
output = encoder(data)
print(output)
print(output.shape)

# Instantiate Patches tokeniser
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

# Instantiate or linear
flat = Flatten()
lin = Linear(25, 10, True)
output = lin(output)
print(output)
print(output.shape)

# Softmax
# soft = Softmax(dim=10)
# output = soft(output)
# print(output)
# print(output.shape)

# Build model
resformer = Sequential(resnet18_last_layer_removed,
                        encoder,
                        lin)



"""
    T R A I N N I N G
"""
train_loader, test_loader = load_intel_scene_data()

resformer.train()

def validation_step(model, batch):
    images, labels = batch 
    images, labels = images.cuda(), labels.cuda()
    out = model.forward(images)
    cross_entropy = nn.CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: nn.Module, val_set: DataLoader):
    outputs = [validation_step(model, batch) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def train(epochs_no, model: nn.Module, train_set: DataLoader, val_set: DataLoader):
    history = []
    
    # TODO Read about optimizer optimizer = opt_func(model.parameters(), lr)
    
    for epoch in range(epochs_no):
        """  Training Phase """ 
        for batch in train_set:
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()
            curr_loss = loss(model.forward(inputs), labels)
            curr_loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
        """ Validation phase """
        result = evaluate(model, val_set)
        history.append(result)
        if epoch % 10 == 0 :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': curr_loss,
                }, PATH)
    return history

def train_model(epochs_no, model_to_train, model_name, dataset):
    device = get_default_device()

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    model = to_device(model_to_train, device)

    train(epochs_no, model, train_loader, val_loader)


train_model(5, resformer)

