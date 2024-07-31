import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm


def register_with_mask_harmonized_original_resolution_exptoinsp():
    bone_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_harmonized"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_STANDARD"

    bone_harm_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/harmonized_BONE_lung_masks"
    std_non_harm_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_lung_masks/expiratory_STANDARD"

    out_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/harmonized/SyN_exptoinsp"

    jd_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/harmonized/jacobian_determinant_exptoinsp_harmonized"
    log_jd_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/harmonized/logjacobian_determinant_exptoinsp_harmonized"

    os.makedirs(out_harm_exp_toinsp, exist_ok=True)
    os.makedirs(jd_harm_exp_toinsp, exist_ok=True)
    os.makedirs(log_jd_harm_exp_toinsp, exist_ok=True)

    bone_harm_files = sorted(os.listdir(bone_harm))
    std_non_harm_files = sorted(os.listdir(std_non_harm))

    bone_harm_masks_files = sorted(os.listdir(bone_harm_masks)) 
    #Grab the files after the 19th file
    bone_harm_masks_files = bone_harm_masks_files[19:]
    print(bone_harm_masks_files)
    std_non_harm_masks_files = sorted(os.listdir(std_non_harm_masks))
    #Grab the files after the 19th file
    std_non_harm_masks_files = std_non_harm_masks_files[19:]
    print(std_non_harm_masks_files)

    for i in tqdm(range(len(bone_harm_files))):
        bone_harm_file = os.path.join(bone_harm, bone_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_harm_mask_file = os.path.join(bone_harm_masks, bone_harm_masks_files[i])
        std_non_harm_mask_file = os.path.join(std_non_harm_masks, std_non_harm_masks_files[i])

        out_harm_exp_toinsp_file = os.path.join(out_harm_exp_toinsp, std_non_harm_files[i])

        jd_harm_exp_toinsp_file = os.path.join(jd_harm_exp_toinsp, std_non_harm_files[i])
        log_jd_harm_exp_toinsp_file = os.path.join(log_jd_harm_exp_toinsp, std_non_harm_files[i])

        #Load the images
        bone_harm_img = ants.image_read(bone_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        bone_harm_mask_img = ants.image_read(bone_harm_mask_file)
        std_non_harm_mask_img = ants.image_read(std_non_harm_mask_file)

        #Run the registration
        reg_exp_to_insp = ants.registration(fixed = bone_harm_img, moving = std_non_harm_img, type_of_transform = 'SyN', mask= bone_harm_mask_img, moving_mask = std_non_harm_mask_img)
        warped_img_exp_to_insp = ants.apply_transforms(fixed = bone_harm_img, moving = std_non_harm_img, transformlist = reg_exp_to_insp['fwdtransforms'])
        jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(bone_harm_img, reg_exp_to_insp['fwdtransforms'][0])
        log_jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(bone_harm_img, reg_exp_to_insp['fwdtransforms'][0], do_log=True)

        #Save the images
        ants.image_write(warped_img_exp_to_insp, out_harm_exp_toinsp_file)
        ants.image_write(jacobian_determinant_exp_to_insp, jd_harm_exp_toinsp_file)
        ants.image_write(log_jacobian_determinant_exp_to_insp, log_jd_harm_exp_toinsp_file)


register_with_mask_harmonized_original_resolution_exptoinsp()