import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import voxelmorph as vxm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from dataloader import VoxelMorphDataloader
import time
import pandas as pd

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

#Dataloader for VoxelMorph
train_dataframe = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled")
valid_dataframe = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled")

train_dataset = VoxelMorphDataloader(train_dataframe)
valid_dataset = VoxelMorphDataloader(valid_dataframe)


#Configure parameters to train VoxelMorph 
batch_size = 4
epochs = 30 
IN_SHAPE = (192,192,192)

enc_nf = [16, 32, 32, 32]
dec_nf =  [32, 32, 32, 32, 32, 16, 16]

#Get gpu ids
gpu_ids = [0,1]

image_loss_func=vxm.losses.NCC().loss 
losses=[image_loss_func]
weights=[1]

losses += [vxm.losses.Grad('l2',loss_mult=2).loss]
weights += [0.01]

#Initialize model
voxmorph = vxm.networks.VxmDense(inshape = IN_SHAPE, nb_unet_features = [enc_nf, dec_nf]) 

optimizer = torch.optim.Adam(voxmorph.parameters(), lr=1e-4)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(gpu_ids) > 1:
    voxmorph = torch.nn.DataParallel(voxmorph)
    voxmorph.save = voxmorph.module.save

voxmorph.to(device)

best_val_loss = float('inf')
train_loss = []
val_loss = []
# for epoch in range(epochs):
#     train_loss = []
#     val_loss = []
#     voxmorph.train()
       
#     print(f"Epoch {epoch}")
#     for i, data in enumerate(train_dataloader):
#         moving, fixed = data
#         moving = moving.to(device)
#         fixed = fixed.to(device)
#         optimizer.zero_grad()
#         loss = voxmorph(moving, fixed)
#         loss.backward()
#         optimizer.step()
#         print(f"Batch {i} Loss {loss.item()}")

#     for i, data in enumerate(valid_dataloader):
#         moving, fixed = data
#         moving = moving.to(device)
#         fixed = fixed.to(device)
#         loss = voxmorph(moving, fixed)
#         print(f"Validation Batch {i} Loss {loss.item()}")

#     voxmorph.save(f'./experiments/epoch_{epoch}.pth')
    # print(f"Model saved at epoch {epoch}")
 
