import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from dataloader import VoxelMorphDataloader
import time
import pandas as pd
import random


#Dataloader for VoxelMorph
train_dataframe = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled")
valid_dataframe = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled")

train_dataset = VoxelMorphDataloader(train_dataframe)
valid_dataset = VoxelMorphDataloader(valid_dataframe)

def plot_utils(moving, fixed, registered):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # moving image
    ax[0].imshow(moving[0, :, :, 96], cmap='gray')
    ax[0].set_title('moving image')
    ax[0].axis('off')

    # fixed image
    ax[1].imshow(fixed[0, :, :, 96], cmap='gray')
    ax[1].set_title('fixed image')
    ax[1].axis('off')

    plt.show()

#Configure parameters to train VoxelMorph 
batch_size = 4
epochs = 30 
IN_SHAPE = (192,192,192)

enc_nf = [16, 32, 32, 32]
dec_nf =  [32, 32, 32, 32, 32, 16, 16]

#Get gpu ids
gpu_ids = [0]

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

def seed_everything(seed=123):
    """
    Seed the experiment
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything()

#Train the model
epoch_loss_final = []
epoch_val_loss_final = []
inference_dir = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/voxelmorph_non_harmonized"
best_val_loss = float('inf')
 
for epoch in range(epochs):
    start_time = time.time() 
    voxmorph.train()
    epoch_loss = []
    epoch_total_loss = [] 
    val_loss = []

    for inputs, truth in enumerate(train_dataloader):
        inputs = [i.to(device).float() for i in inputs]
        truth = [t.to(device).float() for t in truth]
        pred, flow = voxmorph(inputs[0], inputs[1], registration=True)

        loss=0
        loss_list=[]
        for n,loss_function in enumerate(losses):
            curr_loss=loss_function(truth[0],[0])*weights[n]
            loss_list.append(curr_loss.item())
            loss+=curr_loss
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss_final.append(np.mean(epoch_total_loss))

    #Validation 
    with torch.no_grad():
        voxmorph.eval()
        epoch_dir = os.path.join(inference_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        voxmorph.save(os.path.join(epoch_dir, f'epoch_{epoch}.pth'))
        for inputs, truth in enumerate(valid_dataloader):
            inputs = [i.to(device).float() for i in inputs]
            truth = [t.to(device).float() for t in truth]
            pred, flow = voxmorph(inputs[0], inputs[1], registration=True)

            loss=0
            loss_list=[]
            for n,loss_function in enumerate(losses):
                curr_loss=loss_function(truth[0],[0])*weights[n]
                loss_list.append(curr_loss.item())
                loss+=curr_loss
            val_loss.append(loss.item())
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(inputs[0][0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Moving Image')
            plt.subplot(1, 3, 2)
            plt.imshow(inputs[1][0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Fixed Image')
            plt.subplot(1, 3, 3)
            plt.imshow(pred[0][0].detach().cpu().numpy().squeeze(), cmap='gray')
            plt.title('Registered Image')
            plt.savefig(os.path.join(epoch_dir, f'CTreg_{epoch}.png'))  # Save the plot as an image
            plt.close()

        epoch_val_loss_final.append(np.mean(val_loss))
    mean_val_loss = np.mean(val_loss)
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
    end_time = time.time()  # End time
    epoch_time = end_time - start_time  # Time taken for the epoch
    print('------------------------------------------------------')
    print(f'Epoch: {epoch} Loss: {np.mean(epoch_total_loss)}')
    print(f'Epoch {epoch} Val Loss: {np.mean(val_loss)}')
    print(f'Epoch {epoch} Losses: {np.mean(epoch_loss, axis=0)}')
    print(f'Time taken for epoch: {epoch_time} seconds')
    print('------------------------------------------------------')
        

