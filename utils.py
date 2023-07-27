from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR


train_transforms = A.Compose([
    A.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    #A.ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.1, rotate_limit = 15,p=0.4),
    #A.HorizontalFlip(),
    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True,p=0.5),
    A.RandomCrop(height=32, width=32, always_apply=True,p=0.5,),
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
    ToTensorV2()
])

#Test Phase transformations
test_transforms = A.Compose([
                           
                             A.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                             ToTensorV2()

class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
# Model Scheduler
model = resnet().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.03,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
Lr_Finder=LRFinder(model,optimizer,criterion,device='cuda')



model = resnet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
EPOCHS = 20

scheduler = OneCycleLR(optimizer,
                       max_lr=5.70E-02,
                       steps_per_epoch=len(train_loader),
                       epochs=EPOCHS,
                       pct_start=5/EPOCHS,
                       div_factor=100,
                       three_phase=False,
                       final_div_factor=100,
                       anneal_strategy='linear')


# Gradcam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

