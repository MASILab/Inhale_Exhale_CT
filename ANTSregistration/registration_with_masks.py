import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm

def register_with_mask_non_harmonized():
    bone_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_BONE"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_STANDARD"

    bone_res_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_resampled_lung_masks/inspiratory_BONE"
    std_res_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_resampled_lung_masks/expiratory_STANDARD"

    out_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/SyN_exptoinsp"
    # out_non_harm_insptoexp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/SyN_insptoexp_BONE_non_harmonized" 

    jd_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/jacobian_determinant_exptoinsp_nonharmonized"
    log_jd_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/logjacobian_determinant_exptoinsp_nonharmonized"

    os.makedirs(out_non_harm_exp_toinsp, exist_ok=True)
    os.makedir(jd_non_harm_exp_toinsp, exist_ok=True)
    os.makedirs(log_jd_non_harm_exp_toinsp, exist_ok=True)

    # jd_non_harm_insptoexp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/jacobian_determinant_insptoexp_nonharmonized"
    # log_jd_non_harm_insptoexp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/logjacobian_determinant_insptoexp_nonharmonized"

    bone_non_harm_files = sorted(os.listdir(bone_non_harm))
    std_non_harm_files = sorted(os.listdir(std_non_harm))

    bone_res_mask_files = sorted(os.listdir(bone_res_mask))
    std_res_mask_files = sorted(os.listdir(std_res_mask))

    for i in tqdm(range(len(bone_non_harm_files))):
        bone_non_harm_file = os.path.join(bone_non_harm, bone_non_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_res_mask_file = os.path.join(bone_res_mask, bone_res_mask_files[i])
        std_res_mask_file = os.path.join(std_res_mask, std_res_mask_files[i])

        out_non_harm_exp_toinsp_file = os.path.join(out_non_harm_exp_toinsp, std_non_harm_files[i])

        jd_non_harm_exp_toinsp_file = os.path.join(jd_non_harm_exp_toinsp, std_non_harm_files[i])
        log_jd_non_harm_exp_toinsp_file = os.path.join(log_jd_non_harm_exp_toinsp, std_non_harm_files[i])

        # jd_non_harm_insptoexp_file = os.path.join(jd_non_harm_insptoexp, bone_non_harm_files[i])
        # log_jd_non_harm_insptoexp_file = os.path.join(log_jd_non_harm_insptoexp, bone_non_harm_files[i])

        #Load the images
        bone_non_harm_img = ants.image_read(bone_non_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        bone_res_mask_img = ants.image_read(bone_res_mask_file)
        std_res_mask_img = ants.image_read(std_res_mask_file)

        #Run the registration
        reg_exp_to_insp = ants.registration(fixed = std_non_harm_img, moving = bone_non_harm_img, type_of_transform = 'SyN', moving_mask= bone_res_mask_img, mask = std_res_mask_img)
        # warped_img_exp_to_insp = ants.apply_transforms(fixed = bone_non_harm_img, moving = std_non_harm_img, transformlist = reg_exp_to_insp['fwdtransforms'])
        # jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(bone_non_harm_img, reg_exp_to_insp['fwdtransforms'][0])
        # log_jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(bone_non_harm_img, reg_exp_to_insp['fwdtransforms'][0], do_log=True)
        
        # warped_img_insp_to_exp = ants.apply_transforms(fixed = std_non_harm_img, moving = bone_non_harm_img, transformlist = reg_exp_to_insp['invtransforms'])
        jacobian_determinant_insp_to_exp = ants.create_jacobian_determinant_image(std_non_harm_img, reg_exp_to_insp['fwdtransforms'][0])
        log_jacobian_determinant_insp_to_exp = ants.create_jacobian_determinant_image(std_non_harm_img, reg_exp_to_insp['fwdtransforms'][0], do_log=True)

        #Save the images
        # ants.image_write(warped_img_exp_to_insp, out_non_harm_exp_toinsp_file)
        # ants.image_write(jacobian_determinant_exp_to_insp, jd_non_harm_exp_toinsp_file)
        # ants.image_write(log_jacobian_determinant_exp_to_insp, log_jd_non_harm_exp_toinsp_file)

        # ants.image_write(warped_img_insp_to_exp, out_non_harm_insptoexp_file)
        ants.image_write(jacobian_determinant_insp_to_exp, jd_non_harm_insptoexp_file)
        ants.image_write(log_jacobian_determinant_insp_to_exp, log_jd_non_harm_insptoexp_file)


def register_with_mask_harmonized():
    bone_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_STANDARD"

    bone_harm_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/resampled_harmonized_BONE_lung_masks"
    std_non_harm_masks = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/inference_resampled_lung_masks/expiratory_STANDARD"

    out_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/SyN_exptoinsp_STANDARD_harmonized"
    out_harm_insptoexp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/SyN_insptoexp_BONE_harmonized"

    jd_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/jacobian_determinant_exptoinsp_harmonized"
    log_jd_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/logjacobian_determinant_exptoinsp_harmonized"
    
    jd_harm_insptoexp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/jacobian_determinant_insptoexp_harmonized"
    log_jd_harm_insptoexp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/final_regsitered_data_with_masks/logjacobian_determinant_exptoinsp_harmonized"

    bone_harm_files = sorted(os.listdir(bone_harm))
    std_non_harm_files = sorted(os.listdir(std_non_harm))

    bone_harm_masks_files = sorted(os.listdir(bone_harm_masks))
    std_non_harm_masks_files = sorted(os.listdir(std_non_harm_masks))

    for i in tqdm(range(len(bone_harm_files))):
        bone_harm_file = os.path.join(bone_harm, bone_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_harm_mask_file = os.path.join(bone_harm_masks, bone_harm_masks_files[i])
        std_non_harm_mask_file = os.path.join(std_non_harm_masks, std_non_harm_masks_files[i])

        out_harm_exp_toinsp_file = os.path.join(out_harm_exp_toinsp, std_non_harm_files[i])
        out_harm_insptoexp_file = os.path.join(out_harm_insptoexp, bone_harm_files[i])

        jd_harm_exp_toinsp_file = os.path.join(jd_harm_exp_toinsp, std_non_harm_files[i])
        log_jd_harm_exp_toinsp_file = os.path.join(log_jd_harm_exp_toinsp, std_non_harm_files[i])

        jd_harm_insptoexp_file = os.path.join(jd_harm_insptoexp, bone_harm_files[i])
        log_jd_harm_insptoexp_file = os.path.join(log_jd_harm_insptoexp, bone_harm_files[i])

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

        reg_insp_to_exp = ants.registration(fixed = std_non_harm_img, moving = bone_harm_img, type_of_transform = 'SyN', mask= std_non_harm_mask_img, moving_mask = bone_harm_mask_img)
        warped_img_insp_to_exp = ants.apply_transforms(fixed = std_non_harm_img, moving = bone_harm_img, transformlist = reg_insp_to_exp['fwdtransforms'])
        jacobian_determinant_insp_to_exp = ants.create_jacobian_determinant_image(std_non_harm_img, reg_insp_to_exp['fwdtransforms'][0])
        log_jacobian_determinant_insp_to_exp = ants.create_jacobian_determinant_image(std_non_harm_img, reg_insp_to_exp['fwdtransforms'][0], do_log=True)

        #Save the images
        ants.image_write(warped_img_exp_to_insp, out_harm_exp_toinsp_file)
        ants.image_write(jacobian_determinant_exp_to_insp, jd_harm_exp_toinsp_file)
        ants.image_write(log_jacobian_determinant_exp_to_insp, log_jd_harm_exp_toinsp_file)

        ants.image_write(warped_img_insp_to_exp, out_harm_insptoexp_file)
        ants.image_write(jacobian_determinant_insp_to_exp, jd_harm_insptoexp_file)
        ants.image_write(log_jacobian_determinant_insp_to_exp, log_jd_harm_insptoexp_file)


# register_with_mask_non_harmonized()
# register_with_mask_harmonized()


def register_with_mask_non_harmonized_original_resolution_exptoinsp():
    bone_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_BONE"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/images/clipped_masked_out_STANDARD"

    bone_res_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_lung_masks/inspiratory_BONE"
    std_res_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/data/masks/inference_lung_masks/expiratory_STANDARD"

    out_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/SyN_exptoinsp"

    jd_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/jacobian_determinant_exptoinsp_nonharmonized"
    log_jd_non_harm_exp_toinsp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/logjacobian_determinant_exptoinsp_nonharmonized"

    os.makedirs(out_non_harm_exp_toinsp, exist_ok=True)
    os.makedirs(jd_non_harm_exp_toinsp, exist_ok=True)
    os.makedirs(log_jd_non_harm_exp_toinsp, exist_ok=True)


    bone_non_harm_files = sorted(os.listdir(bone_non_harm))
    #Get the 19th file
    bone_non_harm_files = bone_non_harm_files[19:]
    std_non_harm_files = sorted(os.listdir(std_non_harm))
    #Get the 19th file
    std_non_harm_files = std_non_harm_files[19:]

    bone_res_mask_files = sorted(os.listdir(bone_res_mask))
    #get the 19th file
    bone_res_mask_files = bone_res_mask_files[19:]
    std_res_mask_files = sorted(os.listdir(std_res_mask))
    #get the 19th file
    std_res_mask_files = std_res_mask_files[19:]

    print(bone_non_harm_files)
    print(std_non_harm_files)
    print(bone_res_mask_files)
    print(std_res_mask_files)

    for i in tqdm(range(len(bone_non_harm_files))):
        bone_non_harm_file = os.path.join(bone_non_harm, bone_non_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_res_mask_file = os.path.join(bone_res_mask, bone_res_mask_files[i])
        std_res_mask_file = os.path.join(std_res_mask, std_res_mask_files[i])

        out_non_harm_exp_toinsp_file = os.path.join(out_non_harm_exp_toinsp, std_non_harm_files[i])

        jd_non_harm_exp_toinsp_file = os.path.join(jd_non_harm_exp_toinsp, std_non_harm_files[i])
        log_jd_non_harm_exp_toinsp_file = os.path.join(log_jd_non_harm_exp_toinsp, std_non_harm_files[i])

        # jd_non_harm_insptoexp_file = os.path.join(jd_non_harm_insptoexp, bone_non_harm_files[i])
        # log_jd_non_harm_insptoexp_file = os.path.join(log_jd_non_harm_insptoexp, bone_non_harm_files[i])

        #Load the images
        bone_non_harm_img = ants.image_read(bone_non_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        bone_res_mask_img = ants.image_read(bone_res_mask_file)
        std_res_mask_img = ants.image_read(std_res_mask_file)

        #Run the registration
        reg_exp_to_insp = ants.registration(fixed = bone_non_harm_img, moving = std_non_harm_img, type_of_transform = 'SyN', mask= bone_res_mask_img, moving_mask = std_res_mask_img)
        warped_img_exp_to_insp = ants.apply_transforms(fixed = bone_non_harm_img, moving = std_non_harm_img, transformlist = reg_exp_to_insp['fwdtransforms'])
        jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(bone_non_harm_img, reg_exp_to_insp['fwdtransforms'][0])
        log_jacobian_determinant_exp_to_insp = ants.create_jacobian_determinant_image(bone_non_harm_img, reg_exp_to_insp['fwdtransforms'][0], do_log=True)

        #Save the images
        ants.image_write(warped_img_exp_to_insp, out_non_harm_exp_toinsp_file)
        ants.image_write(jacobian_determinant_exp_to_insp, jd_non_harm_exp_toinsp_file)
        ants.image_write(log_jacobian_determinant_exp_to_insp, log_jd_non_harm_exp_toinsp_file)
        print("Image saved! Moving to the next subject.")


register_with_mask_non_harmonized_original_resolution_exptoinsp()

# register_with_mask_harmonized_original_resolution_exptoinsp()