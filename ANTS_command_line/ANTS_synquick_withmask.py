import numpy as np 
from tqdm import tqdm
import subprocess

#Perform the registration using the ANTS command line 
bone_fixed = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/images/clipped_masked_out_BONE/masked_COPDGene_D36309_BONE.nii.gz"
std_moving = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/images/clipped_masked_out_STANDARD/masked_COPDGene_D36309_STANDARD.nii.gz"

bone_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/masks/inference_lung_masks/inspiratory_BONE/COPDGene_D36309_BONE.nii.gz"
std_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/masks/inference_lung_masks/expiratory_STANDARD/COPDGene_D36309_STANDARD.nii.gz"

prefix = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/registration_trial_SyNquick_with_mask/deformable_reg_"



#Use the ANTS command line to register the images

ants_command = f"""
antsRegistration \
--dimensionality 3 \
--output [{prefix},{prefix}Warped.nii.gz, {prefix}InverseWarped.nii.gz] \
--interpolation BSpline \
--transform SyN[0.1,3,0] \
--metric CC[{bone_fixed}, {std_moving}, 1, 4] \
--convergence [100x70x50x0,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--masks [{bone_mask}, {std_mask}]
"""

# ants_command = f"antsRegistration -d 3 -f {bone_fixed} -m {std_moving} -o {prefix} -t s -r 4 -z 1"

subprocess.run(ants_command, shell=True, check=True)