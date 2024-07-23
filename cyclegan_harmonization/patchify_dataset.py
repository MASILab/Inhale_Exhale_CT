import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from PIL import Image
import nibabel as nib


#load a NIfTI slice and plot it 
img = nib.load("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train_slices/inspiratory_BONE/COPDGene_H16787_BONE_211.nii.gz").get_fdata()

#plot the image
plt.imshow(np.rot90(img[:,:,0]), cmap = 'gray', vmin = -1000, vmax=0)
plt.show()