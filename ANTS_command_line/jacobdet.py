import os
import numpy as np
from tqdm import tqdm


non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/nonharm_ANTS_outputs_exptoinsp_large_images" 
harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_ANTS_outputs_exptoinsp_large_images"

#Compute log jacobian determinant using ANTS 

for file in tqdm(os.listdir(non_harmonized)):
    files = os.listdir(os.path.join(non_harmonized, file))
    for f in files:
        if f.endswith("_1InverseWarp.nii.gz"):
            warp_field = os.path.join(non_harmonized, file, f)
    
    output = os.path.join(non_harmonized, file, "inverse_jacobian_det.nii.gz")
    command = f"CreateJacobianDeterminantImage 3 {warp_field} {output} 0 1"
    print(command)
    os.system(command)