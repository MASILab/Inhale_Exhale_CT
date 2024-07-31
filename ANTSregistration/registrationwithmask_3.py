import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm


def register_with_mask_non_harmonized_original_resolution_insptoexp():
    bone_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_BONE"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_STANDARD"

    bone_res_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_lung_masks/inspiratory_BONE"
    std_res_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_lung_masks/expiratory_STANDARD"

    out_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/SyN_insptoexp/"

    jd_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/jacobian_determinant_insptoexp/"
    log_jd_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/log_jacobian_determinant_insptoexp/"

    print(out_non_harm_exp_toinsp)
    print(jd_non_harm_exp_toinsp)
    print(log_jd_non_harm_exp_toinsp)
    os.makedirs(out_non_harm_exp_toinsp, exist_ok=True)
    os.makedirs(jd_non_harm_exp_toinsp, exist_ok=True)
    os.makedirs(log_jd_non_harm_exp_toinsp, exist_ok=True)


    bone_non_harm_files = sorted(os.listdir(bone_non_harm))
    std_non_harm_files = sorted(os.listdir(std_non_harm))
 

    bone_res_mask_files = sorted(os.listdir(bone_res_mask))
    std_res_mask_files = sorted(os.listdir(std_res_mask))

    print(bone_non_harm_files)
    print(std_non_harm_files)
    print(bone_res_mask_files)
    print(std_res_mask_files)

    for i in tqdm(range(len(bone_non_harm_files))):
        bone_non_harm_file = os.path.join(bone_non_harm, bone_non_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_res_mask_file = os.path.join(bone_res_mask, bone_res_mask_files[i])
        std_res_mask_file = os.path.join(std_res_mask, std_res_mask_files[i])

        out_non_harm_exp_toinsp_file = os.path.join(out_non_harm_exp_toinsp, bone_non_harm_files[i])

        jd_non_harm_exp_toinsp_file = os.path.join(jd_non_harm_exp_toinsp, bone_non_harm_files[i])
        log_jd_non_harm_exp_toinsp_file = os.path.join(log_jd_non_harm_exp_toinsp, bone_non_harm_files[i])

        # jd_non_harm_insptoexp_file = os.path.join(jd_non_harm_insptoexp, bone_non_harm_files[i])
        # log_jd_non_harm_insptoexp_file = os.path.join(log_jd_non_harm_insptoexp, bone_non_harm_files[i])

        #Load the images
        bone_non_harm_img = ants.image_read(bone_non_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        bone_res_mask_img = ants.image_read(bone_res_mask_file)
        std_res_mask_img = ants.image_read(std_res_mask_file)

        #Run the registration
        reg_exp_to_insp = ants.registration(moving = bone_non_harm_img, fixed = std_non_harm_img, type_of_transform = 'SyN', moving_mask= bone_res_mask_img, mask = std_res_mask_img)
        warped_img_exp_to_insp = ants.apply_transforms(fixed = std_non_harm_img, moving = bone_non_harm_img, transformlist = reg_exp_to_insp['fwdtransforms'])
        jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(std_non_harm_img, reg_exp_to_insp['fwdtransforms'][0])
        log_jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(std_non_harm_img, reg_exp_to_insp['fwdtransforms'][0], do_log=True)

        #Save the images
        ants.image_write(warped_img_exp_to_insp, out_non_harm_exp_toinsp_file)
        ants.image_write(jacobian_determinant_exp_to_insp, jd_non_harm_exp_toinsp_file)
        ants.image_write(log_jacobian_determinant_exp_to_insp, log_jd_non_harm_exp_toinsp_file)
        print("Image saved! Moving to the next subject.")


register_with_mask_non_harmonized_original_resolution_insptoexp()