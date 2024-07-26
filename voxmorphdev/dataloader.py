import nibabel as nib 
import numpy as np
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch' 
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from scipy.interpolate import interp1d



class VoxelMorphDataloader(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.normalizer = interp1d([-1024, 0], [0, 1])

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.loc[idx]
        pid = row['FileName']
        insp_bone = row['inspiratory_BONE']
        exp_std = row['expiratory_STANDARD']

        bone_tensor = self.normalize(insp_bone)
        std_tensor = self.normalize(exp_std)

        #Voxelmorph expects 2 channels where the moving and fixed images will be concatenated
        bone_tensor = bone_tensor.unsqueeze(0).float()
        std_tensor = std_tensor.unsqueeze(0).float()

        return [std_tensor, bone_tensor], [bone_tensor] #Return a lits of the moving tesnor and fixed tensor followed by the fixed tensor


    def normalize(self, data):
        img = nib.load(data).get_fdata()
        clipped_img = np.clip(img, -1024, 0)
        norm_img = self.normalizer(clipped_img)
        torch_tensor = torch.from_numpy(norm_img).float()
        return torch_tensor
 

 #Test dataloader 
# train_df_harm = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/test_data_voxelmorph.csv")
# val_df_val = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/valid_data_voxelmorph.csv")

# train_dataset = VoxelMorphDataloader(train_df_harm)
# valid_dataset = VoxelMorphDataloader(val_df_val)

# train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=6)
# valid_loader = DataLoader(dataset=valid_dataset, batch_size=4, shuffle=False, num_workers=6)

# for i, (data, target) in enumerate(train_loader):
#     print("Moving image:", data[0].shape)
#     print("Fixed image:", data[1].shape)
#     print("Target image:", target[0].shape)
#     break
