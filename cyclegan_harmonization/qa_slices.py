import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from tqdm import tqdm   
import nibabel as nib 
from scipy.interpolate import interp1d 
import concurrent.futures

train_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train_slices/expiratory_STANDARD"
images = os.listdir(train_bone)
norm = interp1d([-1024, 3072], [-1,1])
out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/qa_train_slices/STANDARD"


def process_image(file):
    fname = file.split(".nii.gz")[0]
    img = nib.load(os.path.join(train_bone, file)).get_fdata()[:,:,0]
    plt.figure(figsize=(20,10),facecolor='w')
    fig, ax = plt.subplots(1,2)
    clipped = np.clip(img, -1024, 3072)
    normalized = norm(clipped)
    im1 = ax[0].imshow(np.rot90(clipped), cmap='gray', vmin=-1024, vmax=3072)
    ax[0].set_title("Original Image")
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    im2 = ax[1].imshow(np.rot90(normalized), cmap='gray', vmin=-1, vmax=1)
    ax[1].set_title("Normalized Image")
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(out_path, fname + ".png"), format='png', dpi=300)
    plt.ioff()
    plt.clf()
    plt.close('all')

with concurrent.futures.ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, images), total=len(images)))


