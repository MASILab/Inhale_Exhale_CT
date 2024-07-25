import numpy as np 
import SimpleITK as sitk
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

"""
#Script to resample masked out images to isotropic spatial resolution before performing affine/rigid registration of images
"""

def resample_image(image, new_size=(192, 192, 192)):
    # Calculate the original size and spacing
    original_size = np.array(image.GetSize(), dtype=int)
    original_spacing = np.array(image.GetSpacing(), dtype=float)

    # Calculate the new spacing based on the original size and new size
    new_spacing = (original_size * original_spacing) / new_size
    new_spacing = [float(spacing) for spacing in new_spacing]

    # Resample using SimpleITK
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkBSpline)

    resampled_image = resample.Execute(image)
    return resampled_image


def clip_image(image, min_value=-1024, max_value=0):
    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Clip the values in the numpy array
    clipped_array = np.clip(image_array, min_value, max_value)
    
    # Convert the numpy array back to a SimpleITK image
    clipped_image = sitk.GetImageFromArray(clipped_array)
    
    # Preserve the orientation, origin, and spacing
    clipped_image.SetDirection(image.GetDirection())
    clipped_image.SetOrigin(image.GetOrigin())
    clipped_image.SetSpacing(image.GetSpacing())
    return clipped_image



def resample_dataset():
    data_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/original"

    #Reasmple to isotropic resolution (which is what we want)
    valid_bone = os.path.join(data_path, "valid_masked_out_BONE")
    valid_std = os.path.join(data_path, "valid_masked_out_STANDARD")

    test_bone = os.path.join(data_path, "test_masked_out_BONE")
    test_std = os.path.join(data_path, "test_masked_out_STANDARD")

    output_bone_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_BONE"
    output_std_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    output_bone_test = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_BONE"
    output_std_test = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD"

    os.makedirs(output_bone_val, exist_ok = True)
    os.makedirs(output_std_val, exist_ok = True)

    os.makedirs(output_bone_test, exist_ok = True)
    os.makedirs(output_std_test, exist_ok = True)

    #Validation
    bone_files = os.listdir(valid_bone)
    std_files = os.listdir(valid_std)

    #Test
    bone_files_test = os.listdir(test_bone)
    std_files_test = os.listdir(test_std)


    for i in tqdm(range(len(bone_files))):
        img = sitk.ReadImage(os.path.join(valid_bone, bone_files[i]))
        resampled_img = resample_image(img)
        clipped_img = clip_image(resampled_img) 
        output_path = os.path.join(output_bone_val, f"resampled_{bone_files[i]}")
        sitk.WriteImage(clipped_img, output_path)


    for i in tqdm(range(len(std_files))):
        img = sitk.ReadImage(os.path.join(valid_std, std_files[i]))
        resampled_img = resample_image(img)
        clipped_img = clip_image(resampled_img)
        output_path = os.path.join(output_std_val, f"resampled_{std_files[i]}")
        sitk.WriteImage(clipped_img, output_path)


    for i in tqdm(range(len(bone_files_test))):
        img = sitk.ReadImage(os.path.join(test_bone, bone_files_test[i]))
        resampled_img = resample_image(img)
        clipped_img = clip_image(resampled_img)

        output_path = os.path.join(output_bone_test, f"resampled_{bone_files_test[i]}")
        sitk.WriteImage(clipped_img, output_path)


    for i in tqdm(range(len(std_files_test))):
        img = sitk.ReadImage(os.path.join(test_std, std_files_test[i]))
        resampled_img = resample_image(img)
        clipped_img = clip_image(resampled_img)

        output_path = os.path.join(output_std_test, f"resampled_{std_files_test[i]}")
        sitk.WriteImage(clipped_img, output_path)


def clip_intensities():
    output_bone_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_BONE"
    output_std_val = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_val_masked_out_STANDARD"

    output_bone_test = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_BONE"
    output_std_test = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_test_masked_out_STANDARD"

    bone_files = os.listdir(output_bone_val)
    std_files = os.listdir(output_std_val)

    bone_files_test = os.listdir(output_bone_test)
    std_files_test = os.listdir(output_std_test)

    for i in tqdm(range(len(bone_files))):
        img = sitk.ReadImage(os.path.join(output_bone_val, bone_files[i]))
        clipped_img = clip_image(img)
        sitk.WriteImage(clipped_img, os.path.join(output_bone_val, f"clipped_{bone_files[i]}"))
    
    for i in tqdm(range(len(std_files))):
        img = sitk.ReadImage(os.path.join(output_std_val, std_files[i]))
        clipped_img = clip_image(img)
        sitk.WriteImage(clipped_img, os.path.join(output_std_val, f"clipped_{std_files[i]}"))
    
    for i in tqdm(range(len(bone_files_test))):
        img = sitk.ReadImage(os.path.join(output_bone_test, bone_files_test[i]))
        clipped_img = clip_image(img)
        sitk.WriteImage(clipped_img, os.path.join(output_bone_test, f"clipped_{bone_files_test[i]}"))
    
    for i in tqdm(range(len(std_files_test))):
        img = sitk.ReadImage(os.path.join(output_std_test, std_files_test[i]))
        clipped_img = clip_image(img)
        sitk.WriteImage(clipped_img, os.path.join(output_std_test, f"clipped_{std_files_test[i]}"))


def resample_harmonized_images():
    path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/original"
    output_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled"

    os.makedirs(output_path, exist_ok = True)

    files = os.listdir(path)

    for i in tqdm(range(len(files))):
        img = sitk.ReadImage(os.path.join(path, files[i]))
        resampled_img = resample_image(img)
        clipped_img = clip_image(resampled_img)
        sitk.WriteImage(clipped_img, os.path.join(output_path, f"resampled_{files[i]}"))


def resample_harmonized_val_images():
    path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/original/val_masked_out_harmonized"
    output_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/resampled_val_masked_out_harmonized"

    os.makedirs(output_path, exist_ok = True)

    files = os.listdir(path)

    for i in tqdm(range(len(files))):
        img = sitk.ReadImage(os.path.join(path, files[i]))
        resampled_img = resample_image(img)
        clipped_img = clip_image(resampled_img)
        sitk.WriteImage(clipped_img, os.path.join(output_path, f"resampled_{files[i]}"))


# resample_harmonized_images()

resample_harmonized_val_images()

# clip_intensities()



# resample_dataset()
