import os
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

jd_exp_insp = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_original_resolution/non_harmonized/logjacobian_determinant_exptoinsp_nonharmonized"

files = os.listdir(jd_exp_insp)

controls = [file for file in files if file.endswith("_control.nii.gz")]
cases = [file for file in files if not file.endswith("_control.nii.gz")]

print("Number of controls: ", len(controls))
print("Number of cases: ", len(cases))

def get_log_jacobian(file_list):
    for file in tqdm(file_list):
        img = nib.load(os.path.join(jd_exp_insp, file))
        data = img.get_fdata()
        yield data.flatten()

plt.figure(figsize=(20, 20), facecolor='w')

# Plot density for each control map
for jacobians in get_log_jacobian(controls):
    sns.kdeplot(jacobians, bw_adjust=0.5, label='Control', color='blue', alpha=0.5)

# Plot density for each case map
for jacobians in get_log_jacobian(cases):
    sns.kdeplot(jacobians, bw_adjust=0.5, label='Case', color='orange', alpha=0.5)

# Add labels and legend
plt.xlabel('Jacobian Values')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.title('Density Plot of Jacobian Determinants for Cases and Controls')
plt.savefig("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/SPIE_paper_figures/jacobian_determinant_cases_vs_controls.tiff", dpi=300)
plt.close()