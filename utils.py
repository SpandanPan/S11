from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from resnet import ResNet18

# Defining the train and test data
train_transforms = A.Compose([
    A.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    A.PadIfNeeded(min_height=48, min_width=48, always_apply=True,p=0.5),
    A.RandomCrop(height=32, width=32, always_apply=True,p=0.5,),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
    ToTensorV2()
])

#Test Phase transformations
test_transforms = A.Compose([
                             A.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                             ToTensorV2()])



class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

train_data = Cifar10SearchDataset(root='./data', train=True,download=True, transform=train_transforms)
test_data = Cifar10SearchDataset(root='./data', train=False,download=True, transform=test_transforms)


# Device
SEED = 1
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)


# Defining Train and Test Loader
dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
# test dataloader
test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)



classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
# Model Scheduler
#model = ResNet18().to(device)
#optimizer=optim.Adam(model.parameters(),lr=0.03,weight_decay=1e-4)
#criterion = nn.CrossEntropyLoss()
#Lr_Finder=LRFinder(model,optimizer,criterion,device='cuda')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.CrossEntropyLoss()

# Define test and train functions

train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct,512,
        100. * correct / 512))

from torch.optim.lr_scheduler import OneCycleLR
EPOCHS=5
scheduler = OneCycleLR(optimizer,
                       max_lr=5.70E-02,
                       steps_per_epoch=512,
                       epochs=EPOCHS,
                       pct_start=2/EPOCHS,
                       div_factor=100,
                       three_phase=False,
                       final_div_factor=100,
                       anneal_strategy='linear')

def misclassified_image(test_loader,device,model,train_data):

  cnt=0
  data, target = next(iter(test_loader))
  data, target = data.to(device), target.to(device)
  output = model(data)
  pred = output.argmax(dim=1, keepdim=True)
  img_lst=[]
  img_tensor=[]
  cat_lst=[]
  x_lst=[]
  for i in range(0,127):
    x = random.randint(0,127)

    if pred[x].item()!=target[x].item():
      img1=data[x]
      img=data[x].permute(1, 2, 0).cpu().numpy()
      img_lst.append(img)
      img_tensor.append(img1)
      cat=[value for key,value in enumerate(train_data.class_to_idx) if key==target[x].item()][0]
      cat_lst.append(cat)
      x_lst.append(x)
      cnt+=1
    if cnt>9:
      break
  return img_lst,img_tensor,cat_lst


invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def grad_cam(model):
    target_layers = [model.layer4[-1]]
    placeholder=torch.zeros(size=(3,3,32,32))
    input_tensor=placeholder.cuda()
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0,:]
    return grayscale_cam
