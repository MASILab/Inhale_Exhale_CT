import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm
import matplotlib.pyplot as plt

#Need to register inpisratory to expiratory image and compute the jacobian determinant of 
#Non harmonized data: Inspiratory to expiratory 
#Harmonized data: Inspiratory to expiratory 

#better to do a mask based registration. 

def register_no_mask_non_harmonized():
    #Register non harmonized
    bone_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_BONE"
    std_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_STANDARD" 

    out_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/SyN_registered_BONE_inspiratory_images_without_mask"

    jd_non_hamrm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/jacobian_determinant_images_insptoexp_withoutmask"
    log_jd_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/log_jacobian_determinant_images_insptoexp_withoutmask"


    bone_non_harm_files = sorted(os.listdir(bone_non_harmonized))
    std_non_harm_files = sorted(os.listdir(std_non_harmonized))

    #Run ANTS registration on the non harmonized images
    for i in tqdm(range(len(bone_non_harm_files))):
        bone_non_harm_file = os.path.join(bone_non_harmonized, bone_non_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harmonized, std_non_harm_files[i])

        out_non_harm_file = os.path.join(out_non_harmonized, bone_non_harm_files[i])

        #Load the images
        #Run ANTS registration
        bone_non_harm_img = ants.image_read(bone_non_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        #Run the registration
        reg = ants.registration(fixed = std_non_harm_img, moving = bone_non_harm_img, type_of_transform = 'SyN')
        warped_img = ants.apply_transforms(fixed = std_non_harm_img, moving = bone_non_harm_img, transformlist = reg['fwdtransforms'])
        jacobian_determinant = ants.create_jacobian_determinant_image(std_non_harm_img, reg['fwdtransforms'][0])
        log_jacobian_determinant = ants.create_jacobian_determinant_image(std_non_harm_img, reg['fwdtransforms'][0], do_log=True)

        #Save the images
        ants.image_write(warped_img, out_non_harm_file)
        ants.image_write(jacobian_determinant, os.path.join(jd_non_hamrm, bone_non_harm_files[i]))
        ants.image_write(log_jacobian_determinant, os.path.join(log_jd_non_harm, bone_non_harm_files[i]))


def register_without_mask_harmonized():
    std_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_STANDARD"
    bone_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled" 

    out_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/SyN_registered_BONE_inspiratory_images_without_mask"

    jd_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/jacobian_determinant_images_insptoexp_withoutmask"
    log_jd_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/log_jacobian_determinant_images_insptoexp_withoutmask"

    bone_harm_files = sorted(os.listdir(bone_harmonized))
    std_non_harm_files = sorted(os.listdir(std_non_harmonized))

    #Run ANTS registration on the harmonized images 
    for i in tqdm(range(len(bone_harm_files))):
        bone_harm_file = os.path.join(bone_harmonized, bone_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harmonized, std_non_harm_files[i])

        out_harm_file = os.path.join(out_harmonized, bone_harm_files[i])

        #Load the images
        #Run ANTS registration
        bone_harm_img = ants.image_read(bone_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        #Run the registration
        reg = ants.registration(fixed = std_non_harm_img, moving = bone_harm_img, type_of_transform = 'SyN')
        warped_img = ants.apply_transforms(fixed = std_non_harm_img, moving = bone_harm_img, transformlist = reg['fwdtransforms'])
        jacobian_determinant = ants.create_jacobian_determinant_image(std_non_harm_img, reg['fwdtransforms'][0])
        log_jacobian_determinant = ants.create_jacobian_determinant_image(std_non_harm_img, reg['fwdtransforms'][0], do_log=True)

        #Save the images
        ants.image_write(warped_img, out_harm_file)
        ants.image_write(jacobian_determinant, os.path.join(jd_harm, bone_harm_files[i]))
        ants.image_write(log_jacobian_determinant, os.path.join(log_jd_harm, bone_harm_files[i]))


# register_no_mask_non_harmonized()

# register_without_mask_harmonized()

def compute_log_jacobian_determinant_non_harmonized_exptoinsp_reg():
    bone_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_BONE"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_STANDARD"

    logjd_out_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/log_jacobian_determinant_images_exptoinsp_reg"

    bone_non_harm_files = sorted(os.listdir(bone_non_harm))
    std_non_harm_files = sorted(os.listdir(std_non_harm))

    for i in tqdm(range(len(bone_non_harm_files))):
        bone_non_harm_file = os.path.join(bone_non_harm, bone_non_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_non_harm_img = ants.image_read(bone_non_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        reg = ants.registration(fixed = bone_non_harm_img, moving = std_non_harm_img, type_of_transform = 'SyN')
        log_jacobian_determinant = ants.create_jacobian_determinant_image(bone_non_harm_img, reg['fwdtransforms'][0], do_log=True)

        ants.image_write(log_jacobian_determinant, os.path.join(logjd_out_harm, std_non_harm_files[i]))


def compute_log_jacobian_determinant_harmonized_exptoinsp_reg():
    bone_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled"
    std_non_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_STANDARD"

    logjd_out_harm = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/log_jacobian_determinant_images_exptoinsp_reg"

    bone_harm_files = sorted(os.listdir(bone_harm))
    std_non_harm_files = sorted(os.listdir(std_non_harm))

    for i in tqdm(range(len(bone_harm_files))):
        bone_harm_file = os.path.join(bone_harm, bone_harm_files[i])
        std_non_harm_file = os.path.join(std_non_harm, std_non_harm_files[i])

        bone_harm_img = ants.image_read(bone_harm_file)
        std_non_harm_img = ants.image_read(std_non_harm_file)

        reg = ants.registration(fixed = bone_harm_img, moving = std_non_harm_img, type_of_transform = 'SyN')
        log_jacobian_determinant = ants.create_jacobian_determinant_image(bone_harm_img, reg['fwdtransforms'][0], do_log=True)

        ants.image_write(log_jacobian_determinant, os.path.join(logjd_out_harm, std_non_harm_files[i]))

compute_log_jacobian_determinant_non_harmonized_exptoinsp_reg()
compute_log_jacobian_determinant_harmonized_exptoinsp_reg()

