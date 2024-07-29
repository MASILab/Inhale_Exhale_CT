import pandas as pd
from Emphysemamodel.lungmask import ProcessLungMask
import logging
import os
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger()


class GetLungMask:
    def __init__(self, in_ct_dir, project_dir):
        self.in_ct_dir = in_ct_dir
        self.project_dir = project_dir

    def _generate_lung_mask_config(self):
        return {
            'input': {
                'ct_dir': self.in_ct_dir
            },
            'output': {
                'root_dir': self.project_dir,
                'if_overwrite': True
            },
            'model': {
                'model_lung_mask': '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/models/lung_mask' #Must add this model to my location in /nfs/masi/krishar1
            }
        }

    def generate_lung_mask(self):
        """
        Preprocessing, generating masks, level prediction, get the TCI evaluation, etc.
        :return:
        """
        # logger.info(f'##### Start preprocess #####')
        config_preprocess = self._generate_lung_mask_config()
        logger.info(f'Get lung mask\n')
        lung_mask_generator = ProcessLungMask(config_preprocess)
        lung_mask_generator.run()


paths = ["/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train/inspiratory_BONE", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train/expiratory_STANDARD"]
val_paths = ["/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/inspiratory_BONE", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/expiratory_STANDARD"]
test_paths = ["/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/inspiratory_BONE", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/expiratory_STANDARD"]

harmonized_data = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results/insp_exp_COPD_epoch5_run1_outputs"
out_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/harmonized_epoch5"

harmonized_val_data = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/insp_exp_COPD_epoch5_val"
harmonized_val_mask_out = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/lungmasks/harmonized_epoch5_val"

# for file_path in harmonized_val_data:
reg_paths = ["/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/SyN_registered_STANDARD_expiratory_images", 
             "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/SyN_registered_BONE_inspiratory_images_without_mask", 
             "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/SyN_registered_BONE_inspiratory_images_without_mask", 
             "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/SyN_registered_STANDARD_expiratory_images"]


for file_path in tqdm(reg_paths):
    intctdir = file_path
    outct_dir = os.path.join(file_path + "_lungmask")
    os.makedirs(outct_dir, exist_ok=True)
    lungmask = GetLungMask(intctdir, outct_dir)
    lungmask.generate_lung_mask()
