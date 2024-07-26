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
from tqdm import tqdm
from scipy.interpolate import interp1d
import wandb

#Dataloader for VoxelMorph
train_dataframe = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/train_data_voxelmorph.csv")
valid_dataframe = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/valid_data_voxelmorph.csv")

train_dataset = VoxelMorphDataloader(train_dataframe)
valid_dataset = VoxelMorphDataloader(valid_dataframe)


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

to_hu = interp1d([0,1], [-1024, 0])

#Initialize model
voxmorph = vxm.networks.VxmDense(inshape = IN_SHAPE, nb_unet_features = [enc_nf, dec_nf]) 

optimizer = torch.optim.Adam(voxmorph.parameters(), lr=1e-4)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(gpu_ids) > 1:
    voxmorph = torch.nn.DataParallel(voxmorph)
    voxmorph.save = voxmorph.module.save

print("Number of parameters in VoxelMorph:", sum([p.numel() for p in voxmorph.parameters() if p.requires_grad]))

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

run = wandb.init(

    project="Voxelmorph_lung_registration",   
    name="COPD_non_harmonized_deformable_registration_L2loss=0.01",
    config={
        "learning_rate": 0.0001,
        "epochs": 30,
        "Loss fn": "NCC, L2_norm",
        "Optimizer":"Adam",
        "NCC Hyperparameters":1,
        "L2_norm Hyperparameters":0.01,
        "Batch Size":4,
        "NCC_Win_Size":9,
        "Epochs storage":f'{inference_dir}'
    },
)
 
for epoch in tqdm(range(epochs)):
    print(f"Start training voxelmorph. Current epoch is {epoch}")
    start_time = time.time() 
    voxmorph.train()
    epoch_loss = []
    epoch_total_loss = [] 
    val_loss = []

    for i, (inputs, truth) in enumerate(tqdm(train_dataloader)):
        #Dataloader returns a list of moving and fixed images followed by the list of the fixed image
        inputs = [i.to(device).float() for i in inputs]
        truth = [t.to(device).float() for t in truth]
        pred, flow = voxmorph(inputs[0], inputs[1], registration=True)

        loss=0
        loss_list=[]
        for n,loss_function in enumerate(losses):
            curr_loss=loss_function(truth[0],pred)*weights[n]
            loss_list.append(curr_loss.item())
            loss+=curr_loss
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss_final.append(np.mean(epoch_total_loss))

    
    mean_epoch_loss = np.mean(epoch_loss, axis=0)

    #Validation 
    with torch.no_grad():
        voxmorph.eval()
        print(f"Validating model for epoch {epoch}")
        epoch_dir = os.path.join(inference_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        voxmorph.save(os.path.join(epoch_dir, f'epoch_{epoch}.pth'))
        for i, (inputs, truth) in enumerate(tqdm(valid_dataloader)):
            inputs = [i.to(device) for i in inputs]
            truth = [t.to(device) for t in truth]
            val_pred, val_flow = voxmorph(inputs[0], inputs[1], registration=True)

            loss=0
            loss_list=[]
            for n,loss_function in enumerate(losses):
                curr_loss=loss_function(truth[0], pred)*weights[n]
                loss_list.append(curr_loss.item())
                loss+=curr_loss
            val_loss.append(loss.item())
            
            plt.figure(figsize=(20, 5))
            plt.subplot(2, 2, 1)
            moving_img = inputs[0][0].cpu().numpy().squeeze() #Image still in the range of 0-1
            clipped_moving = np.clip(moving_img, 0, 1)
            mid_moving = moving_img.shape[2] // 2
            plt.imshow(np.rot90(to_hu(clipped_moving[:,:,mid_moving])), cmap='gray', vmin=-1024, vmax=0)
            plt.colorbar()
            plt.title('Moving expiratory immage')

            plt.subplot(2, 2, 2)
            fixed_img = inputs[1][0].cpu().numpy().squeeze()
            clipped_fixed = np.clip(fixed_img, 0, 1)
            mid_fixed = fixed_img.shape[2] // 2
            plt.imshow(np.rot90(to_hu(clipped_fixed[:,:,mid_fixed])), cmap='gray', vmin=-1024, vmax=0)
            plt.colorbar()
            plt.title('Fixed inspiratory image')

            plt.subplot(2, 2, 3)
            flow_np = val_flow[0].detach().cpu().numpy()
            mid_slice = flow_np.shape[3] // 2  # Middle slice along the depth axis
            z_displacement = flow_np[2, :, :, mid_slice]
            plt.imshow(np.rot90(z_displacement), cmap='jet')
            plt.colorbar()
            plt.title('Z Displacement field')

            plt.subplot(2, 2, 4)
            prediction = val_pred[0][0].detach().cpu().numpy().squeeze()
            clipped_prediction = np.clip(prediction, 0, 1)
            mid_pred = prediction.shape[2] // 2
            plt.imshow(np.rot90(to_hu(clipped_prediction[:,:,mid_pred])), cmap='gray', vmin=-1024, vmax=0)
            plt.colorbar()
            plt.title('Registered expiratory image')
            plt.savefig(os.path.join(epoch_dir, f'CTreg_{epoch}{i}.png'))  # Save the plot as an image
            plt.close()

        epoch_val_loss_final.append(np.mean(val_loss))
    mean_val_loss = np.mean(val_loss)
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
    end_time = time.time()  # End time
    epoch_time = end_time - start_time  # Time taken for the epoch

    wandb.log({"Epoch": epoch, 
               "Train Loss": np.mean(epoch_total_loss), 
               "Val Loss": np.mean(val_loss), "NCC Loss": mean_epoch_loss[0], 
                "Smoohtness Loss": mean_epoch_loss[1]})


    print('------------------------------------------------------')
    print(f'Epoch: {epoch} Loss: {np.mean(epoch_total_loss)}')
    print(f'Epoch {epoch} Val Loss: {np.mean(val_loss)}')
    print(f'Epoch {epoch} Losses: {np.mean(epoch_loss, axis=0)}')
    print(f'Time taken for epoch: {epoch_time} seconds')
    print('------------------------------------------------------')
        

