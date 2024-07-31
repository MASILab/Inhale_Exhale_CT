import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm

jd_exp_insp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/SyN_exptoinsp"

files = os.listdir(jd_exp_insp)

controls = [file for file in files if file.endswith("_control.nii.gz")]
cases = [file for file in files if not file.endswith("_control.nii.gz")]

print("Number of controls: ", len(controls))
print("Number of cases: ", len(cases))

#Create a density plot with the x axis being log jacobian and y axis being normalized density. Cases and control should be in one plot with different colors

def get_log_jacobian(file_list):
    log_jacob_values = []
    for file in tqdm(file_list):
        img = nib.load(os.path.join(jd_exp_insp, file))
        data = img.get_fdata()
        flattened = data.flatten()

        return flattened

control_jacobians = get_log_jacobian(controls)
case_jacobians = get_log_jacobian(cases)

plt.figure(figsize=(10, 6), facecolor='w')
plt.hist(control_jacobians, bins=100, density=True, alpha=0.5, label='Controls', color='blue')
plt.hist(case_jacobians, bins=100, density=True, alpha=0.5, label='Cases', color='orange')

# Add labels and legend
plt.xlabel('Jacobian Values')
plt.ylabel('Normalized Voxel Count')
plt.legend(loc='upper right')
plt.title('Histogram of Jacobian Determinants for Cases and Controls')
plt.savefig("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/SPIE_paper_figures/jacobian_determinant_cases_vs_controls.tiff", dpi=300)

# Show plot
plt.show()


