import os
import numpy as np
from tqdm import tqdm


#apply deformation fields to the lung masks to get the deformed masks 
# fixed_inhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/insp_BONE_emphysema/emphysema"
fixed_inhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/harmonized_emphysema_epoch5/emphysema"
moving_exhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/exp_STANDARD_emphysema/emphysema"

# non_harmonized_transforms = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized"

harmonized_transforms = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exptoinsp_harmonized"

# output_lung_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized_emphysema/warped_emphysema_insp_to_exp" 
output_lung_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exptoinsp_harmonized_emphysema/warped_emphysema_insp_to_exp"

fixed_masks = sorted(os.listdir(fixed_inhalation_masks))
moving_masks = sorted(os.listdir(moving_exhalation_masks))
transforms = sorted(os.listdir(harmonized_transforms))

for fixed, moving, transform in tqdm(zip(fixed_masks, moving_masks, transforms), total=len(fixed_masks)):
    fixed_mask_path = os.path.join(fixed_inhalation_masks, fixed)
    moving_mask_path = os.path.join(moving_exhalation_masks, moving)
    transform_path = os.path.join(harmonized_transforms, transform)

    files_transforms = os.listdir(transform_path)
    for file in files_transforms:
        if file.endswith("_1InverseWarp.nii.gz"):
            warp_field = os.path.join(harmonized_transforms, transform, file)
        if file.endswith("_0GenericAffine.mat"):
            affine_field = os.path.join(harmonized_transforms, transform, file)
        if file.endswith("_InverseWarped.nii.gz"):
            outfile = file

    output_mask_path = os.path.join(output_lung_masks, outfile)

    print("Deforming the mask for ", moving)
    print("Apply the ANTS transforms")

    #Apply the transform to the moving mask
    # command = f"antsApplyTransforms -d 3 -i {moving_mask_path} -r {fixed_mask_path} -t {warp_field} -t {affine_field} -n NearestNeighbor -o {output_mask_path}"
    command = f"antsApplyTransforms -d 3 -i {fixed_mask_path} -r {moving_mask_path} -t {warp_field} -t {[ affine_field, 1 ]} -n NearestNeighbor -o {output_mask_path}"
    print(command)
    os.system(command)