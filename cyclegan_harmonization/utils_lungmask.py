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

for file_path in paths:
    intctdir = file_path
    outct_dir = file_path.replace("train", "lungmasks")
    os.makedirs(outct_dir, exist_ok=True)
    lungmask = GetLungMask(intctdir, outct_dir)
    lungmask.generate_lung_mask()
