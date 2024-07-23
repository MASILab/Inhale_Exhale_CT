import nibabel as nib 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd 
from scipy.interpolate import interp1d



class VoxelMorphDataloader(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.normalizer = interp1d([-1024, 3072], [0, 1])

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.loc[idx]
        pid = row['FileName']
        insp_bone = row['inspiratory_BONE']
        exp_std = row['expiratory_STANDARD']

        bone_tensor = self.normalize(insp_bone)
        std_tensor = self.normalize(exp_std)

        return [std_tensor, bone_tensor], [bone_tensor]


    def normalize(self, data):
        img = nib.load(data).get_fdata()
        clipped_img = np.clip(img, -1024, 3072)
        norm_img = self.normalizer(clipped_img)
        torch_tensor = torch.from_numpy(norm_img).float()
        return torch_tensor
 