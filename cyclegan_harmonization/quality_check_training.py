import nibabel as nib 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from PIL import Image 
import os
from scipy.interpolate import interp1d

#Write a loop to generate images for all possible epochs for different conversions.
def convert_toHU(normalized_slice):
    scale_range = [-1,1]
    clip_range = [-1024, 3072]

    in_slice = np.clip(normalized_slice, scale_range[0], scale_range[1])
    normalizer = interp1d(scale_range, clip_range)
    hu_slice = normalizer(in_slice)
    hu_slice = np.clip(hu_slice, clip_range[0], clip_range[1])
    return hu_slice

real = Image.open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/inspiration_expiration_COPD/images/epoch030_real_A.png").convert('L')
fake = Image.open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/inspiration_expiration_COPD/images/epoch030_fake_B.png").convert('L')


#convert the PIL images into numpy arrays 
real_arr = np.array(real).astype(np.float32) #Images are saved in the range of 0-255 while training.
fake_arr = np.array(fake).astype(np.float32)

real_arrHU = (real_arr/255) * (3072 - (-1024)) + (-1024) #Using the transform only for the purpose of checking if the model is learning correctly or not
fake_arrHU = (fake_arr/255) * (3072 - (-1024)) + (-1024)

diff = real_arrHU - fake_arrHU

fig = plt.figure(figsize=(25,25))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
im1 = ax1.imshow(np.rot90(real_arrHU),cmap = "gray", vmin = -1024, vmax = 600)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title("Real inspiratory image")
im2 = ax2.imshow(np.rot90(fake_arrHU), cmap = 'gray', vmin = -1024, vmax = 600)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title("Synthetic inspiratory image")
im3 = ax3.imshow(np.rot90(diff), cmap = "gray", vmin = -1024, vmax = 600)
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title("Difference Image")
plt.show()
