import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from Models.resnet import ResNet18
from utils import train_transforms,test_transforms,Cifar10SearchDataset,scheduler,grayscale_cam,invTrans,misclassified_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data
print('==> Preparing data..')

# Defining the train and test data
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

# Model Summary 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18().to(device)
print(summary(model, input_size=(3, 32, 32)))

# Model Training
EPOCHS=20
model = ResNet18().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.03,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, criterion)
    test(model, device, test_loader)
    scheduler.step()

# Plot train/test loss and accuracy
fig, axs = plt.subplots(2,2, figsize=(20,15))

axs[0,0].set_title('Train Losses')
axs[1,0].set_title('Test Losses')
axs[0,0].plot(train_losses)
axs[1,0].plot(test_losses)

# Plotting wrong predictions
img_lst,img_tensor,cat_lst = misclassified_image(test_loader,device,model,train_data)
for i in range(10):
  plt.subplot(5,2,i+1)
  plt.tight_layout()
  plt.imshow(img_lst[i])
  plt.title(cat_lst[i])
  plt.xticks([])
  plt.yticks([])


# Plotting grad cam for misclassified images
for i in range(10):
  img_req=img_tensor[i].to(device)
  inv_tensor = invTrans(img_req)
  inv_tensor_1=inv_tensor.cpu().permute(1, 2, 0).numpy()
  visualization = show_cam_on_image(inv_tensor_1,grayscale_cam, use_rgb=True)
  plt.subplot(5,2,i+1)
  plt.tight_layout()
  plt.imshow(visualization)
  plt.title(cat_lst[i])
  plt.xticks([])
  plt.yticks([])
