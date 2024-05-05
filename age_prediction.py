# %%
# import
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam import GradCAM
import argparse
import os
import torch
import glob
import re
import torch.nn as nn
import numpy as np
from utils.functions import fix_seed
from utils.datasets import DatasetBCP_T_2D
from utils.train import train
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.Conv2D import Conv2D
from models.Res2D import Res2D
from models.Conv_ASPP import ConvASPP
from models.Conv_Attention import ConvAttention
import nibabel as nib
from nibabel.processing import conform
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random

# %%
# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--modality", type=str, default="T2")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu", type=list, default=[0, 1])
parser.add_argument("--use_amp", type=bool, default=True)  # gradscalar
opt = parser.parse_args([])
print(opt)

# %%
# device
device = torch.device(
    "cuda", opt.gpu[0]) if torch.cuda.is_available() else "cpu"
fix_seed(opt.seed)

# make directory
PATH = "/home/yshuai/age_predict/" + f"{opt.modality}"
os.makedirs(PATH, exist_ok=True)

# %%
# plot distribution
file_paths = sorted(
    glob.glob("/home/yshuai/age_predict/data/BCP/T1/*.nii.gz", recursive=True))
labels1 = []
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    match = re.search(r'_(\d+)', file_name)
    if match:
        label = int(match.group(1))
        labels1.append(label)

file_paths = sorted(
    glob.glob("/home/yshuai/age_predict/data/BCP/T2/*.nii.gz", recursive=True))
labels2 = []
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    match = re.search(r'_(\d+)', file_name)
    if match:
        label = int(match.group(1))
        labels2.append(label)

plt.figure(figsize=(8, 5))
plt.hist(labels1, bins=range(0, 25, 1),
         label='T1', alpha=0.6, edgecolor='black')
plt.hist(labels2, bins=range(0, 25, 1),
         label='T2', alpha=0.8, edgecolor='black')
plt.title('Age Distribution', fontsize=17)
plt.xlabel('Month Age', fontsize=14)
plt.ylabel('Number of scans', fontsize=14)
plt.legend(fontsize=12)
plt.savefig('dist.png')

# %%
# read T2 data
file_paths = sorted(
    glob.glob("/home/yshuai/age_predict/data/BCP/T2/*.nii.gz", recursive=True))
voxels = []
labels = []
print('----- Reading Data -----')
for file_path in tqdm(file_paths):
    voxel = nib.squeeze_image(nib.as_closest_canonical(nib.load(file_path)))
    voxel = conform(voxel, (208, 300, 320),
                    voxel_size=(0.8, 0.8, 0.8), order=1)
    voxel = voxel.get_fdata().astype(np.float32)
    voxels.append(voxel)

    file_name = os.path.basename(file_path)
    match = re.search(r'_(\d+)', file_name)
    if match:
        label = int(match.group(1))
        labels.append(label)

# %%
# dataset
dataset = DatasetBCP_T_2D(voxels, labels, mode='cc')
del voxels, labels
dataset_size = len(dataset)
train_size = int(dataset_size * 0.6)
test_size = dataset_size - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
val_size = int(dataset_size * 0.2)
test_size = test_size - val_size
val_set, test_set = random_split(test_set, [val_size, test_size])

# %%
# dataloader
num_workers = 0
train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# %%
# setup
model = Conv2D(1)
# model = Res2D(1,1,[2,2,2,2])
# model=ConvAttention(1,208,300)
# model=ConvASPP(1,208,300)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.00001)

criterion = nn.L1Loss()
scaler = GradScaler(enabled=opt.use_amp)

# train
print('----- Start Training -----')
best_model_dict = train(PATH, model, optimizer, scheduler, criterion, scaler,
                        device, opt.epochs, train_loader, val_loader, use_amp=opt.use_amp)

# %%
# test
dataset.is_test = True
dataset.mode = 'cc'
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
test_loss = 0
best_model = Conv2D(1)
# best_model=ConvAttention(1,208,300)
# best_model=ConvASPP(1,208,300)
# best_model=Res2D(1,1,[2,2,2,2])

best_model.load_state_dict(torch.load('T2/cnn_cc_model.pth'))
best_model.to(device)
best_model.eval()
criterion = nn.L1Loss()

for images, labels in test_loader:
    images = images.permute(1, 0, 2, 3)
    labels = (labels.unsqueeze(1)).repeat(images.shape[0], 1).squeeze()
    images, labels = images.to(device), labels.to(device)
    images = images.to(device)
    outputs = best_model(images)
    loss = criterion(outputs, labels)
    test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test MAE Loss: {test_loss}')

# %%
# gradcam

dataset.is_test = True
dataset.mode = 'cc'
subject = 17
images, labels = test_set[subject][0].unsqueeze(1), test_set[subject][1]
print(f'Subject age = {labels.item()}')

# plot original image
fig, axs = plt.subplots(1, 8, figsize=(20, 5))
for i in range(8):
    ax = axs[i]
    ax.imshow(
        np.rot90(np.stack([(images[10+4*i][0]).cpu().numpy()]*3, axis=-1)/2+0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'z={170+10+4*i}', fontsize=13)
    if i == 0:
        ax.set_ylabel('Original', fontsize=14)
plt.savefig('original.png')

# plot gradcam image
best_model = Conv2D(1)
best_model.load_state_dict(torch.load('T2/cnn_cc_model.pth'))
images = images.to(device)
target_layers = [best_model.conv5]
targets = [BinaryClassifierOutputTarget(1)]
cam = GradCAM(model=best_model, target_layers=target_layers)
fig, axs = plt.subplots(1, 8, figsize=(20, 5))
for i in range(8):
    ax = axs[i]
    grayscale_cam = cam(
        input_tensor=images[10+4*i].unsqueeze(0), targets=targets)
    visualization = show_cam_on_image(np.stack(
        [(images[10+4*i][0]).cpu().numpy()]*3, axis=-1)/2+0.5, grayscale_cam[0], use_rgb=True)
    ax.imshow(np.rot90(visualization))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'z={170+10+4*i}', fontsize=13)
    if i == 0:
        ax.set_ylabel('CNN', fontsize=14)
plt.savefig('cnn_cc_grad.png')
