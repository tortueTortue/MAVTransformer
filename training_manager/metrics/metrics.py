"""
Here are the the methods computing the metrics used for evaluaton
"""
import torch
from torch.nn import Module

def accuracy(outputs, labels):
    """
    Accuracy = no_of_correct_preidctions / no_of_predictions

    *Note: Use this when the classes have about the amount of occurences.
    """
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def print_accuracy_per_class(model: Module, classes: list, batch_size: int):
    class_amount = len(classes)
    class_correct = list(0. for i in range(class_amount))
    class_total = list(0. for i in range(class_amount))
    with torch.no_grad():
        for batch in test_loader:
            i, l = batch
            i, l = i.cuda(), l.cuda()
            out = model(i)
            _, predicted = torch.max(out, 1)
            c = (predicted == l).squeeze()
            for i in range(batch_size):
                label = l[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(class_amount):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))