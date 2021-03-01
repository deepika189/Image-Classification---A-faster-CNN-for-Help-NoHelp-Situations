from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.utils.prune as prune
import torch.nn.functional as F

plt.ion()


from utils.load_model import load_model
from utils.load_data import load_data
from utils.test_model import test_model
from utils.visualize import imshow, visualize_model

dataloaders, dataset_sizes, class_names = load_data('../data')

model = load_model('resnet18_01')

# test_model(model, dataloaders['test'])

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.7)
#         prune.l1_unstructured(module, name='bias', amount=0.2)

    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)
#         prune.l1_unstructured(module, name='bias', amount=0.4)    

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        prune.remove(module, 'weight')

test_model(model, dataloaders['test'])



torch.save(model.state_dict(), '../models/resnet18_01_pruned')