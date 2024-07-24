import numpy as np 
import SimpleITK as sitk
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

valid_data = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid"
test_data = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test"

#Reasmple to isotropic resolution (which is what we want)
valid_bone = os.path.join(valid_data, "inspiratory_BONE")
valid_std = os.path.join(valid_data, "expiratory_STANDARD")

test_bone = os.path.join(test_data, "inspiratory_BONE")
test_std = os.path.join(test_data, "expiratory_STANDARD")

output_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/inspiratory_BONE"
output_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/expiratory_STANDARD"

output_bone_resize = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/inspiratory_BONE_resized"
output_std_resize = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid_resampled/expiratory_STANDARD_resized"


output_bone_test = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test_resampled/inspiratory_BONE"
output_std_test = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test_resampled/expiratory_STANDARD"

output_bone_test_resize = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test_resampled/inspiratory_BONE_resized"
output_std_test_resize = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test_resampled/expiratory_STANDARD_resized"


os.makedirs(output_bone, exist_ok = True)
os.makedirs(output_std, exist_ok = True)

os.makedirs(output_bone_test, exist_ok = True)
os.makedirs(output_std_test, exist_ok = True)


bone_files = os.listdir(valid_bone)
std_files = os.listdir(valid_std)

bone_files_test = os.listdir(test_bone)
std_files_test = os.listdir(test_std)



def resample_image(image, new_size=(192, 192, 192), new_spacing=(1.0, 1.0, 1.0)):
    # Ensure new_spacing is a list of floats
    new_spacing = [float(spacing) for spacing in new_spacing]

    # Calculate the original size and spacing
    original_size = np.array(image.GetSize(), dtype=int)
    original_spacing = np.array(image.GetSpacing(), dtype=float)

    # Calculate the new size based on the original spacing and new_spacing
    new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()

    # Resample using SimpleITK
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkBSpline)

    resampled_image = resample.Execute(image)
    return resampled_image


def resize_image(image, new_size=(192, 192, 192)):
    # Ensure new_size is a list of integers
    new_size = [int(size) for size in new_size]

    # Resample using SimpleITK
    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkBSpline)

    resampled_image = resample.Execute(image)
    return resampled_image



for i in tqdm(range(len(bone_files))):
    img = sitk.ReadImage(os.path.join(valid_bone, bone_files[i]))
    resampled_img = resample_image(img)

    output_path = os.path.join(output_bone, f"resampled_{bone_files[i]}")
    sitk.WriteImage(resampled_img, output_path)


for i in tqdm(range(len(std_files))):
    img = sitk.ReadImage(os.path.join(valid_std, std_files[i]))
    resampled_img = resample_image(img)

    output_path = os.path.join(output_std, f"resampled_{std_files[i]}")
    sitk.WriteImage(resampled_img, output_path)


for i in tqdm(range(len(bone_files_test))):
    img = sitk.ReadImage(os.path.join(test_bone, bone_files_test[i]))
    resampled_img = resample_image(img)

    output_path = os.path.join(output_bone_test, f"resampled_{bone_files_test[i]}")
    sitk.WriteImage(resampled_img, output_path)


for i in tqdm(range(len(std_files_test))):
    img = sitk.ReadImage(os.path.join(test_std, std_files_test[i]))
    resampled_img = resample_image(img)

    output_path = os.path.join(output_std_test, f"resampled_{std_files_test[i]}")
    sitk.WriteImage(resampled_img, output_path)
