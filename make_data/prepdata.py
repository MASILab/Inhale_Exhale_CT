import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm

#Extract information from the nifti images 
# 1) Extract the convolution kernel from the csv file 
# 2) Extract the image name from the csv file
# 3) Extract the spacing of the image (x,y,z from nibabel)
# 4) Extract the image size (x,y,z from nibabel)
# 5) Extract the number of slices in the image (z dimension)
#6) Extract the orientation from the image (nib affine to ax codes)

nifti_path = "/nfs/masi/krishar1/InhaleExhaleCT_DICOMtoNIfTI/NIfTI"
df = pd.read_csv("/nfs/masi/krishar1/InhaleExhaleCT_DICOMtoNIfTI/PRIVIA_usable_scans.csv")

uids = df['SeriesUID'].to_list()

shape = {}
orientation = {}

for file in tqdm(uids):
    nii = file + ".nii.gz"
    if nii in os.listdir(nifti_path):
        img = nib.load(os.path.join(nifti_path, nii))
        img_data = img.get_fdata()
        img_shape = img_data.shape
        img_spacing = img.header.get_zooms()
        img_orientation = nib.aff2axcodes(img.affine)
        #Add the information to the dataframe in a new column
        df.loc[df['SeriesUID'] == file, 'ImageSize'] = str(img_shape)
        df.loc[df['SeriesUID'] == file, 'Orientation'] = str(img_orientation)

df.to_csv("/nfs/masi/krishar1/InhaleExhaleCT_DICOMtoNIfTI/PRIVIA_usable_scans.csv", index = False)