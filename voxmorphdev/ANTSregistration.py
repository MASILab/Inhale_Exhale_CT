import numpy as np 
import ants
import os
import nibabel as nib 
from tqdm import tqdm
import matplotlib.pyplot as plt

#Affine register the harmonized and non harmonized images 
#Affine register the inspiratory BONE and expiratory STANDARD images for non harmonized data 
#Affine register the inspiratory BONE and expiratory STANDARD images for harmonized data

#Non harmonized data 
def register_non_harmonzied():
    train_data_bone_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_BONE"
    train_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD"

    val_data_bone_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_BONE"
    val_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    bone_files_val = sorted(os.listdir(val_data_bone_non_harmonized))
    standard_files_val = sorted(os.listdir(val_data_standard_non_harmonized))

    bone_files_train = sorted(os.listdir(train_data_bone_non_harmonized))
    standard_files_train = sorted(os.listdir(train_data_standard_non_harmonized))

    output_train = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/train_SyN_registered_STANDARD_expiratory_images"
    # output_train_affine = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/train_affine_matrices"
    output_deform_field = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/train_deform_fields"
    os.makedirs(output_train, exist_ok = True)
    os.makedirs(output_deform_field, exist_ok = True)
    output_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/val_SyN_registered_STANDARD_expiratory_images"
    # output_val_affine = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/val_affine_matrices"
    output_val_deform_field = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/val_deform_fields"
    os.makedirs(output_val, exist_ok = True)
    os.makedirs(output_val_deform_field, exist_ok = True)


    for i in tqdm(range(len(bone_files_train))):
        bone_img = ants.image_read(os.path.join(train_data_bone_non_harmonized, bone_files_train[i]))
        standard_img = ants.image_read(os.path.join(train_data_standard_non_harmonized, standard_files_train[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        warped_image = ants.apply_transforms(fixed = bone_img, moving = standard_img, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(output_train, f"deformable_registered_{standard_files_train[i]}"))

        displacement_field = ants.create_warped_grid(standard_img, transform=affine_reg['fwdtransforms'][0], fixed_reference_image=bone_img)
        displacement_array = displacement_field.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Plot the middle slice in the x-direction
        axes[0].imshow(displacement_array[displacement_array.shape[0] // 2, :, :], cmap='gray')
        axes[0].set_title('Displacement Field (X Slice)')

        # Plot the middle slice in the y-direction
        axes[1].imshow(displacement_array[:, displacement_array.shape[1] // 2, :], cmap='gray')
        axes[1].set_title('Displacement Field (Y Slice)')

        # Plot the middle slice in the z-direction
        axes[2].imshow(displacement_array[:, :, displacement_array.shape[2] // 2], cmap='gray')
        axes[2].set_title('Displacement Field (Z Slice)')

        plt.savefig(os.path.join(output_val_deform_field, f"deform_field_{standard_files_train[i].split('.')[0]}.png"))
        plt.close()



    for i in tqdm(range(len(bone_files_val))):
        bone_img = ants.image_read(os.path.join(val_data_bone_non_harmonized, bone_files_val[i]))
        standard_img = ants.image_read(os.path.join(val_data_standard_non_harmonized, standard_files_val[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        warped_image = ants.apply_transforms(fixed = bone_img, moving = standard_img, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(output_val, f"deformable_registered_{standard_files_val[i]}"))

        displacement_field = ants.create_warped_grid(standard_img, transform=affine_reg['fwdtransforms'][0], fixed_reference_image=bone_img)
        displacement_array = displacement_field.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Plot the middle slice in the x-direction
        axes[0].imshow(displacement_array[displacement_array.shape[0] // 2, :, :], cmap='gray')
        axes[0].set_title('Displacement Field (X Slice)')

        # Plot the middle slice in the y-direction
        axes[1].imshow(displacement_array[:, displacement_array.shape[1] // 2, :], cmap='gray')
        axes[1].set_title('Displacement Field (Y Slice)')

        # Plot the middle slice in the z-direction
        axes[2].imshow(displacement_array[:, :, displacement_array.shape[2] // 2], cmap='gray')
        axes[2].set_title('Displacement Field (Z Slice)')

        plt.savefig(os.path.join(output_deform_field, f"deform_field_{standard_files_val[i].split('.')[0]}.png"))
        plt.close()

 
# def register_harmonized():

# register_non_harmonzied()


def register_harmonized():
    train_data_bone_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_test_masked_out_harmonized"
    train_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD" 

    val_data_bone_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_val_masked_out_harmonized"
    val_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    bone_files_val = sorted(os.listdir(val_data_bone_harmonized))
    standard_files_val = sorted(os.listdir(val_data_standard_non_harmonized))

    bone_files_train = sorted(os.listdir(train_data_bone_harmonized))
    standard_files_train = sorted(os.listdir(train_data_standard_non_harmonized))

    output_train = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/train_SyN_registered_STANDARD_expiratory_images"
    # output_train_affine = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/train_affine_matrices"
    output_deform_field = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/train_deform_fields"
    os.makedirs(output_train, exist_ok = True)
    os.makedirs(output_deform_field, exist_ok = True)
    output_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/val_SyN_registered_STANDARD_expiratory_images"
    # output_val_affine = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/val_affine_matrices"
    output_val_deform_field = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/val_deform_fields"
    os.makedirs(output_val, exist_ok = True)
    os.makedirs(output_val_deform_field, exist_ok = True)


    for i in tqdm(range(len(bone_files_train))):
        bone_img = ants.image_read(os.path.join(train_data_bone_harmonized, bone_files_train[i]))
        standard_img = ants.image_read(os.path.join(train_data_standard_non_harmonized, standard_files_train[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        warped_image = ants.apply_transforms(fixed = bone_img, moving = standard_img, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(output_train, f"deformable_registered_{standard_files_train[i]}"))

        displacement_field = ants.create_warped_grid(standard_img, transform=affine_reg['fwdtransforms'][0], fixed_reference_image=bone_img)
        displacement_array = displacement_field.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Plot the middle slice in the x-direction
        axes[0].imshow(displacement_array[displacement_array.shape[0] // 2, :, :], cmap='gray')
        axes[0].set_title('Displacement Field (X Slice)')

        # Plot the middle slice in the y-direction
        axes[1].imshow(displacement_array[:, displacement_array.shape[1] // 2, :], cmap='gray')
        axes[1].set_title('Displacement Field (Y Slice)')

        # Plot the middle slice in the z-direction
        axes[2].imshow(displacement_array[:, :, displacement_array.shape[2] // 2], cmap='gray')
        axes[2].set_title('Displacement Field (Z Slice)')

        plt.savefig(os.path.join(output_deform_field, f"deform_field_{standard_files_train[i].split('.')[0]}.png"))
        plt.close()
    
    for i in tqdm(range(len(bone_files_val))):
        bone_img = ants.image_read(os.path.join(val_data_bone_harmonized, bone_files_val[i]))
        standard_img = ants.image_read(os.path.join(val_data_standard_non_harmonized, standard_files_val[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        warped_image = ants.apply_transforms(fixed = bone_img, moving = standard_img, transformlist = affine_reg['fwdtransforms'])

        ants.image_write(warped_image, os.path.join(output_val, f"deformable_registered_{standard_files_val[i]}"))

        displacement_field = ants.create_warped_grid(standard_img, transform=affine_reg['fwdtransforms'][0], fixed_reference_image=bone_img)
        displacement_array = displacement_field.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Plot the middle slice in the x-direction
        axes[0].imshow(displacement_array[displacement_array.shape[0] // 2, :, :], cmap='gray')
        axes[0].set_title('Displacement Field (X Slice)')

        # Plot the middle slice in the y-direction
        axes[1].imshow(displacement_array[:, displacement_array.shape[1] // 2, :], cmap='gray')
        axes[1].set_title('Displacement Field (Y Slice)')

        # Plot the middle slice in the z-direction
        axes[2].imshow(displacement_array[:, :, displacement_array.shape[2] // 2], cmap='gray')
        axes[2].set_title('Displacement Field (Z Slice)')

        plt.savefig(os.path.join(output_val_deform_field, f"deform_field_{standard_files_val[i].split('.')[0]}.png"))
        plt.close()

# register_harmonized()


def save_jacobian_determinant_image_non_harmonized():
    train_data_bone_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_BONE"
    train_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD"

    val_data_bone_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_BONE"
    val_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    bone_files_val = sorted(os.listdir(val_data_bone_non_harmonized))
    standard_files_val = sorted(os.listdir(val_data_standard_non_harmonized))

    bone_files_train = sorted(os.listdir(train_data_bone_non_harmonized))
    standard_files_train = sorted(os.listdir(train_data_standard_non_harmonized)) 

    output_train = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/train_jacobian_determinant_images"
    os.makedirs(output_train, exist_ok = True)

    output_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/val_jacobian_determinant_images"
    os.makedirs(output_val, exist_ok = True)

    for i in tqdm(range(len(bone_files_train))):
        bone_img = ants.image_read(os.path.join(train_data_bone_non_harmonized, bone_files_train[i]))
        standard_img = ants.image_read(os.path.join(train_data_standard_non_harmonized, standard_files_train[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        jacobian_determinant = ants.create_jacobian_determinant_image(bone_img, affine_reg['fwdtransforms'][0])

        ants.image_write(jacobian_determinant, os.path.join(output_train, f"jacobian_determinant_{standard_files_train[i]}"))
    
    for i in tqdm(range(len(bone_files_val))):
        bone_img = ants.image_read(os.path.join(val_data_bone_non_harmonized, bone_files_val[i]))
        standard_img = ants.image_read(os.path.join(val_data_standard_non_harmonized, standard_files_val[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        jacobian_determinant = ants.create_jacobian_determinant_image(bone_img, affine_reg['fwdtransforms'][0])

        ants.image_write(jacobian_determinant, os.path.join(output_val, f"jacobian_determinant_{standard_files_val[i]}"))


def save_jacobian_determinant_image_harmonized():
    train_data_bone_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_test_masked_out_harmonized"
    train_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD" 

    val_data_bone_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_val_masked_out_harmonized"
    val_data_standard_non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    bone_files_val = sorted(os.listdir(val_data_bone_harmonized))
    standard_files_val = sorted(os.listdir(val_data_standard_non_harmonized))

    bone_files_train = sorted(os.listdir(train_data_bone_harmonized))
    standard_files_train = sorted(os.listdir(train_data_standard_non_harmonized))

    output_train = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/train_jacobian_determinant_images"
    os.makedirs(output_train, exist_ok = True)

    output_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/val_jacobian_determinant_images"
    os.makedirs(output_val, exist_ok = True)

    for i in tqdm(range(len(bone_files_train))):
        bone_img = ants.image_read(os.path.join(train_data_bone_harmonized, bone_files_train[i]))
        standard_img = ants.image_read(os.path.join(train_data_standard_non_harmonized, standard_files_train[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        jacobian_determinant = ants.create_jacobian_determinant_image(bone_img, affine_reg['fwdtransforms'][0])

        ants.image_write(jacobian_determinant, os.path.join(output_train, f"jacobian_determinant_{standard_files_train[i]}"))
    
    for i in tqdm(range(len(bone_files_val))):
        bone_img = ants.image_read(os.path.join(val_data_bone_harmonized, bone_files_val[i]))
        standard_img = ants.image_read(os.path.join(val_data_standard_non_harmonized, standard_files_val[i]))

        affine_reg = ants.registration(fixed = bone_img, moving = standard_img, type_of_transform = 'SyN')

        jacobian_determinant = ants.create_jacobian_determinant_image(bone_img, affine_reg['fwdtransforms'][0])

        ants.image_write(jacobian_determinant, os.path.join(output_val, f"jacobian_determinant_{standard_files_val[i]}"))


save_jacobian_determinant_image_harmonized()
save_jacobian_determinant_image_non_harmonized() 