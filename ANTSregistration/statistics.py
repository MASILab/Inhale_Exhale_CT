import scipy.stats 
import statsmodels.stats.api as sms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from tqdm import tqdm
import nibabel as nib


non_harmonized_validation = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/val_jacobian_determinant_images"
harmonized_validation = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/val_jacobian_determinant_images"

non_harmonized_validation_files = sorted(os.listdir(non_harmonized_validation))
harmonized_validation_files = sorted(os.listdir(harmonized_validation))

out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/summary_statistics/overlaid_jacobian_determinant_plot"
os.makedirs(out_path, exist_ok=True)

for file in range(len(non_harmonized_validation)):
    non_harmonized_img = nib.load(os.path.join(non_harmonized_validation, non_harmonized_validation_files[file])).get_fdata().flatten()
    harmonized_img = nib.load(os.path.join(harmonized_validation, harmonized_validation_files[file])).get_fdata().flatten()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(non_harmonized_img, bins = 100, color='orange', label='Non-Harmonized', align = 'mid')
    plt.hist(harmonized_img, bins=100, color='blue', label='Harmonized', align = 'mid')
    ax.set_xlabel('Jacobian determinant Values')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,2)
    ax.set_ylim(0,5000000)
    ax.set_title('Histogram of Non-Harmonized and Harmonized Data')
    ax.legend()
    plt.savefig(os.path.join(out_path, f"{non_harmonized_validation_files[file]}_overlaid_histogram.png"))
    plt.close()

