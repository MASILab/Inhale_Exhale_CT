import os
import numpy as np
from tqdm import tqdm


#apply deformation fields to the lung masks to get the deformed masks 
def run_emphysema_warp_SyN():
    subjects_BONE = ["COPDGene_A43240_BONE.nii.gz", "COPDGene_B05639_BONE.nii.gz", "COPDGene_B14644_BONE.nii.gz", "COPDGene_C64585_BONE.nii.gz", 
                     "COPDGene_D48362_BONE.nii.gz", "COPDGene_D70993_BONE.nii.gz", "COPDGene_D80990_BONE_control.nii.gz", "COPDGene_D90444_BONE.nii.gz", 
                     "COPDGene_E59904_BONE.nii.gz", "COPDGene_E69868_BONE.nii.gz", "COPDGene_E73754_BONE_control.nii.gz"]
    
    subjects_STANDARD = ["COPDGene_A43240_STANDARD.nii.gz", "COPDGene_B05639_STANDARD.nii.gz", "COPDGene_B14644_STANDARD.nii.gz", "COPDGene_C64585_STANDARD.nii.gz",
                            "COPDGene_D48362_STANDARD.nii.gz", "COPDGene_D70993_STANDARD.nii.gz", "COPDGene_D80990_STANDARD_control.nii.gz", "COPDGene_D90444_STANDARD.nii.gz",
                            "COPDGene_E59904_STANDARD.nii.gz", "COPDGene_E69868_STANDARD.nii.gz", "COPDGene_E73754_STANDARD_control.nii.gz"]
    

    fixed_inhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/harmonized_emphysema_epoch5/emphysema"

    moving_exhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/exp_STANDARD_emphysema/emphysema"

    harmonized_transforms = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_cases_ANTS_fullrun"
 
    output_lung_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/emphysema_harm_ANTS_fullrun_11subjects"

    fixed_masks = sorted([file for file in os.listdir(fixed_inhalation_masks) if file in subjects_BONE])
    moving_masks = sorted([file for file in os.listdir(moving_exhalation_masks) if file in subjects_STANDARD])
    transforms = sorted(os.listdir(harmonized_transforms))

    for fixed, moving, transform in tqdm(zip(fixed_masks, moving_masks, transforms), total=len(fixed_masks)):
        fixed_mask_path = os.path.join(fixed_inhalation_masks, fixed)
        moving_mask_path = os.path.join(moving_exhalation_masks, moving)
        transform_path = os.path.join(harmonized_transforms, transform)

        files_transforms = os.listdir(transform_path)
        for file in files_transforms:
            if file.endswith("_1Warp.nii.gz"):
                warp_field = os.path.join(harmonized_transforms, transform, file)
            if file.endswith("_0GenericAffine.mat"):
                affine_field = os.path.join(harmonized_transforms, transform, file)
            if file.endswith("_Warped.nii.gz"):
                outfile = file

        output_mask_path = os.path.join(output_lung_masks, outfile)

        print("Deforming the mask for ", moving)
        print("Apply the ANTS transforms")

        #Apply the transform to the moving mask
        command = f"antsApplyTransforms -d 3 -i {moving_mask_path} -r {fixed_mask_path} -t {warp_field} -t {affine_field} -n NearestNeighbor -o {output_mask_path}"
        print(command)
        os.system(command)


def run_emphysema_warp_rigid_reg():
    fixed_inhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/harmonized_emphysema_epoch5/emphysema"
    moving_exhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/exp_STANDARD_emphysema/emphysema"

    harmonized_transforms = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_ANTS_outputs_exptoinsp_large_images_rigid_registered"

    output = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_rigid_registered_emphysema_masks_warped"

    fixed_masks = sorted(os.listdir(fixed_inhalation_masks))
    moving_masks = sorted(os.listdir(moving_exhalation_masks))
    transforms = sorted(os.listdir(harmonized_transforms))

    for fixed, moving, transform in tqdm(zip(fixed_masks, moving_masks, transforms), total=len(fixed_masks)):
        fixed_mask_path = os.path.join(fixed_inhalation_masks, fixed)
        moving_mask_path = os.path.join(moving_exhalation_masks, moving)
        transform_path = os.path.join(harmonized_transforms, transform)

        files_transforms = os.listdir(transform_path)
        for file in files_transforms:
            if file.endswith("_0GenericAffine.mat"):
                affine_field = os.path.join(harmonized_transforms, transform, file)
            if file.endswith("_Warped.nii.gz"):
                outfile = file

        output_mask_path = os.path.join(output, outfile)

        print("Rigid register the masks for", moving)
        print("Apply the ANTS transforms")

        #Apply the transform to the moving mask
        command = f"antsApplyTransforms -d 3 -i {moving_mask_path} -r {fixed_mask_path} -t {affine_field} -n NearestNeighbor -o {output_mask_path}" #For rigid registration
        print(command)
        os.system(command)

# run_emphysema_warp_rigid_reg()
# run_emphysema_warp_SyN()

def run_emphysema_warp_ANTS_full_SyN():
    #Do for harmonized then non harmonized
    fixed_inhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/harmonized_emphysema_epoch5/emphysema"
    moving_exhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/exp_STANDARD_emphysema/emphysema"
    harmonized_transforms = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_ANTS_outputs_exptoinsp_large_images" 

    output_warp_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_ANTS_outputs_exptoinsp_large_images_emphysema_warped"
    os.makedirs(output_warp_harmonized, exist_ok=True)

    required_files = [file for file in os.listdir(harmonized_transforms)]

    #remove the masked word at the beginning of the files in the list comprehesion and replace STANDARD with BONE and also add a .nii.gz at the end
    required_files_fixed = sorted([file.replace("masked_", "").replace("STANDARD", "BONE") + ".nii.gz" for file in required_files])
    required_files_moving = sorted([file.replace("masked_", "") + ".nii.gz" for file in required_files])
    transforms = sorted(os.listdir(harmonized_transforms))

    for fixed, moving, transform in tqdm(zip(required_files_fixed, required_files_moving, transforms), total=len(required_files_fixed)):
        fixed_mask_path = os.path.join(fixed_inhalation_masks, fixed)
        moving_mask_path = os.path.join(moving_exhalation_masks, moving)
        transform_path = os.path.join(harmonized_transforms, transform)

        files_transforms = os.listdir(transform_path)
        for file in files_transforms:
            if file.endswith("_1Warp.nii.gz"):
                warp_field = os.path.join(harmonized_transforms, transform, file)
            if file.endswith("_0GenericAffine.mat"):
                affine_field = os.path.join(harmonized_transforms, transform, file)
            if file.endswith("_Warped.nii.gz"):
                outfile = file

        output_mask_path = os.path.join(output_warp_harmonized, outfile)

        print("Deforming the mask for ", moving)
        print("Apply the ANTS transforms")

        #Apply the transform to the moving mask
        command = f"antsApplyTransforms -d 3 -i {moving_mask_path} -r {fixed_mask_path} -t {warp_field} -t {affine_field} -n NearestNeighbor -o {output_mask_path}"
        print(command)
        os.system(command)


def run_emphysema_warp_ANTS_full_SyN_nonharmonized():
    #Do for harmonized then non harmonized
    fixed_inhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/insp_BONE_emphysema/emphysema"
    moving_exhalation_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/exp_STANDARD_emphysema/emphysema"
    nonharmonized_transforms = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/nonharm_ANTS_outputs_exptoinsp_large_images" 

    output_warp_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/nonharm_ANTS_outputs_exptoinsp_large_images_emphysema_warped"
    os.makedirs(output_warp_harmonized, exist_ok=True)

    required_files = [file for file in os.listdir(nonharmonized_transforms)]

    #remove the masked word at the beginning of the files in the list comprehesion and replace STANDARD with BONE and also add a .nii.gz at the end
    required_files_fixed = sorted([file.replace("masked_", "").replace("STANDARD", "BONE") + ".nii.gz" for file in required_files])
    required_files_moving = sorted([file.replace("masked_", "") + ".nii.gz" for file in required_files])
    transforms = sorted(os.listdir(nonharmonized_transforms))

    for fixed, moving, transform in tqdm(zip(required_files_fixed, required_files_moving, transforms), total=len(required_files_fixed)):
        fixed_mask_path = os.path.join(fixed_inhalation_masks, fixed)
        moving_mask_path = os.path.join(moving_exhalation_masks, moving)
        transform_path = os.path.join(nonharmonized_transforms, transform)

        files_transforms = os.listdir(transform_path)
        for file in files_transforms:
            if file.endswith("_1Warp.nii.gz"):
                warp_field = os.path.join(nonharmonized_transforms, transform, file)
            if file.endswith("_0GenericAffine.mat"):
                affine_field = os.path.join(nonharmonized_transforms, transform, file)
            if file.endswith("_Warped.nii.gz"):
                outfile = file

        output_mask_path = os.path.join(output_warp_harmonized, outfile)

        print("Deforming the mask for ", moving)
        print("Apply the ANTS transforms")

        #Apply the transform to the moving mask
        command = f"antsApplyTransforms -d 3 -i {moving_mask_path} -r {fixed_mask_path} -t {warp_field} -t {affine_field} -n NearestNeighbor -o {output_mask_path}"
        print(command)
        os.system(command)

# run_emphysema_warp_ANTS_full_SyN()
run_emphysema_warp_ANTS_full_SyN_nonharmonized()