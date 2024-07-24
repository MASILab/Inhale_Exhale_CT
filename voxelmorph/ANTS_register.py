import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm


valid_bone_resampled = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/inspiratory_BONE"
valid_std_resampled = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/expiratory_STANDARD"

affine_valid_resampled = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/affine_registered_STANDARD" 

os.makedirs(affine_valid_resampled, exist_ok = True)

valid_bone_resampled_files = sorted(os.listdir(valid_bone_resampled))
valid_std_resampled_files = sorted(os.listdir(valid_std_resampled)) 

# Make a tuple of the two lists
valid_files = list(zip(valid_bone_resampled_files, valid_std_resampled_files))


for index, file in tqdm(enumerate(valid_files)):
    bone_img = ants.image_read(os.path.join(valid_bone_resampled, valid_files[index][0]))
    std_img = ants.image_read(os.path.join(valid_std_resampled, valid_files[index][1]))

    affine_reg = ants.registration(fixed = bone_img, moving = std_img, type_of_transform = 'Affine') 

    warped_std_image = ants.apply_transforms(fixed = bone_img, moving = std_img, transformlist = affine_reg['fwdtransforms'])

    ants.image_write(warped_std_image, os.path.join(affine_valid_resampled, f"warped_{valid_files[index][1]}"))
