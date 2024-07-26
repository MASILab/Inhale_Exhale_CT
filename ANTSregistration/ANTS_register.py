import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm

def mask_out_lung_valid():
    valid_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/inspiratory_BONE"
    valid_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/expiratory_STANDARD"

    valid_bone_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/valid/inspiratory_BONE/lung_mask"
    valid_std_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/valid/expiratory_STANDARD/lung_mask"

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data"

    os.makedirs(masked_out_path, exist_ok = True)

    valid_bone_resampled_files = sorted(os.listdir(valid_bone))
    valid_std_resampled_files = sorted(os.listdir(valid_std)) 

    valid_bone_mask_files = sorted(os.listdir(valid_bone_mask))
    valid_std_mask_files = sorted(os.listdir(valid_std_mask))

    # Make a tuple of the two lists
    valid_files = list(zip(valid_bone_resampled_files, valid_std_resampled_files))
    valid_mask_files = list(zip(valid_bone_mask_files, valid_std_mask_files))


    for index, file in tqdm(enumerate(valid_files)):
        bone_img = nib.load(os.path.join(valid_bone, valid_files[index][0]))
        std_img = nib.load(os.path.join(valid_std, valid_files[index][1]))
        bone_mask = nib.load(os.path.join(valid_bone_mask, valid_mask_files[index][0]))
        std_mask = nib.load(os.path.join(valid_std_mask, valid_mask_files[index][1]))

        bone_img_data = bone_img.get_fdata()
        std_img_data = std_img.get_fdata()
        bone_mask_data = bone_mask.get_fdata()
        std_mask_data = std_mask.get_fdata()

        bone_mask_data[bone_mask_data > 0] = 1
        std_mask_data[std_mask_data > 0] = 1

        bone_img_data = np.multiply(bone_img_data, bone_mask_data)
        std_img_data = np.multiply(std_img_data, std_mask_data)

        bone_img = nib.Nifti1Image(bone_img_data, bone_img.affine)
        std_img = nib.Nifti1Image(std_img_data, std_img.affine)

        nib.save(bone_img, os.path.join(masked_out_path, "valid_masked_out_BONE", f"masked_{valid_files[index][0]}"))
        nib.save(std_img, os.path.join(masked_out_path, "valid_masked_out_STANDARD", f"masked_{valid_files[index][1]}"))


def mask_out_lung_test():
    valid_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/inspiratory_BONE"
    valid_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/expiratory_STANDARD"

    valid_bone_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/test/inspiratory_BONE/lung_mask"
    valid_std_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/test/expiratory_STANDARD/lung_mask"

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/original"

    os.makedirs(masked_out_path, exist_ok = True)

    valid_bone_resampled_files = sorted(os.listdir(valid_bone))
    valid_std_resampled_files = sorted(os.listdir(valid_std)) 

    valid_bone_mask_files = sorted(os.listdir(valid_bone_mask))
    valid_std_mask_files = sorted(os.listdir(valid_std_mask))

    # Make a tuple of the two lists
    valid_files = list(zip(valid_bone_resampled_files, valid_std_resampled_files))
    valid_mask_files = list(zip(valid_bone_mask_files, valid_std_mask_files))


    for index, file in tqdm(enumerate(valid_files)):
        bone_img = nib.load(os.path.join(valid_bone, valid_files[index][0]))
        std_img = nib.load(os.path.join(valid_std, valid_files[index][1]))
        bone_mask = nib.load(os.path.join(valid_bone_mask, valid_mask_files[index][0]))
        std_mask = nib.load(os.path.join(valid_std_mask, valid_mask_files[index][1]))

        bone_img_data = bone_img.get_fdata()
        std_img_data = std_img.get_fdata()
        bone_mask_data = bone_mask.get_fdata()
        std_mask_data = std_mask.get_fdata()

        bone_mask_data[bone_mask_data > 0] = 1
        std_mask_data[std_mask_data > 0] = 1

        bone_img_data = np.multiply(bone_img_data, bone_mask_data)
        std_img_data = np.multiply(std_img_data, std_mask_data)

        bone_img = nib.Nifti1Image(bone_img_data, bone_img.affine)
        std_img = nib.Nifti1Image(std_img_data, std_img.affine)

        nib.save(bone_img, os.path.join(masked_out_path, "test_masked_out_BONE", f"masked_{valid_files[index][0]}"))
        nib.save(std_img, os.path.join(masked_out_path, "test_masked_out_STANDARD", f"masked_{valid_files[index][1]}"))



def mask_out_lung_harmonized():
    harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results/insp_exp_COPD_epoch5_run1_outputs"
    masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/harmonized_epoch5/lung_mask" 

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/original"
    os.makedirs(masked_out_path, exist_ok = True) 

    harmonized_files = sorted(os.listdir(harmonized))
    mask_files = sorted(os.listdir(masks))

    for index, file in tqdm(enumerate(harmonized_files)):
        img = nib.load(os.path.join(harmonized, harmonized_files[index]))
        mask = nib.load(os.path.join(masks, mask_files[index]))

        img_data = img.get_fdata()
        mask_data = mask.get_fdata()

        mask_data[mask_data > 0] = 1

        img_data = np.multiply(img_data, mask_data)

        img = nib.Nifti1Image(img_data, img.affine)

        nib.save(img, os.path.join(masked_out_path, f"masked_{harmonized_files[index]}"))


def mask_out_lung_harmonized_val():
    harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/insp_exp_COPD_epoch5_val"
    masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/harmonized_epoch5_val/lung_mask" 

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/original/val_masked_out_harmonized"
    os.makedirs(masked_out_path, exist_ok = True) 

    harmonized_files = sorted(os.listdir(harmonized))
    mask_files = sorted(os.listdir(masks))

    for index, file in tqdm(enumerate(harmonized_files)):
        img = nib.load(os.path.join(harmonized, harmonized_files[index]))
        mask = nib.load(os.path.join(masks, mask_files[index]))

        img_data = img.get_fdata()

        mask_data = mask.get_fdata()

        mask_data[mask_data > 0] = 1

        img_data = np.multiply(img_data, mask_data)

        img = nib.Nifti1Image(img_data, img.affine)

        nib.save(img, os.path.join(masked_out_path, f"masked_{harmonized_files[index]}"))
    

def affine_register():
    valid_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_BONE"
    valid_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD"

    # valid_bone_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/valid/inspiratory_BONE/lung_mask"
    # valid_std_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/valid/expiratory_STANDARD/lung_mask"

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_rigid_registered_STANDARD"
    os.makedirs(masked_out_path, exist_ok = True)

    valid_bone_files = sorted(os.listdir(valid_bone))
    valid_std_files = sorted(os.listdir(valid_std))

    valid_files = list(zip(valid_bone_files, valid_std_files))


    for index, file in tqdm(enumerate(valid_files)):
        bone = ants.image_read(os.path.join(valid_bone, valid_files[index][0]))
        std = ants.image_read(os.path.join(valid_std, valid_files[index][1]))

        affine_reg = ants.registration(fixed = bone, moving = std, type_of_transform = 'Rigid')

        warped_image = ants.apply_transforms(fixed = bone, moving = std, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(masked_out_path, f"rigid_registered_{valid_files[index][1]}"))


def harmonized_register():
    harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled"
    valid_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD"

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_test_masked_out_rigid_registered_STANDARD"
    os.makedirs(masked_out_path, exist_ok = True)

    harmonized_files = sorted(os.listdir(harmonized))
    valid_std_files = sorted(os.listdir(valid_std))

    valid_files = list(zip(harmonized_files, valid_std_files))

    for index, file in tqdm(enumerate(valid_files)):
        harmonized_img = ants.image_read(os.path.join(harmonized, valid_files[index][0]))
        std = ants.image_read(os.path.join(valid_std, valid_files[index][1]))

        affine_reg = ants.registration(fixed = harmonized_img, moving = std, type_of_transform = 'Rigid')

        warped_image = ants.apply_transforms(fixed = harmonized_img, moving = std, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(masked_out_path, f"rigid_registered_{valid_files[index][1]}"))


def harmonized_register_val():
    harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_val_masked_out_harmonized"
    valid_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    masked_out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled_val_masked_out_rigid_registered_STANDARD"
    os.makedirs(masked_out_path, exist_ok = True)

    harmonized_files = sorted(os.listdir(harmonized))
    valid_std_files = sorted(os.listdir(valid_std))

    valid_files = list(zip(harmonized_files, valid_std_files))

    for index, file in tqdm(enumerate(valid_files)):
        harmonized_img = ants.image_read(os.path.join(harmonized, valid_files[index][0]))
        std = ants.image_read(os.path.join(valid_std, valid_files[index][1]))

        affine_reg = ants.registration(fixed = harmonized_img, moving = std, type_of_transform = 'Rigid')

        warped_image = ants.apply_transforms(fixed = harmonized_img, moving = std, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(masked_out_path, f"rigid_registered_{valid_files[index][1]}"))


harmonized_register_val()
# harmonized_register()
# affine_register()
# mask_out_lung_harmonized()
# mask_out_lung_harmonized_val()
