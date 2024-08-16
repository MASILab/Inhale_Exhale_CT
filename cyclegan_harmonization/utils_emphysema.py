import pandas as pd
from Emphysemamodel.lungmask import ProcessLungMask
import logging
import os
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


logger = logging.getLogger()

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Set GPU ID to 1


class EmphysemaAnalysis:
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

    def get_emphysema_mask(self):
        print(f'Generate emphysema masks')
        lung_mask_dir = os.path.join(self.project_dir, 'lung_mask')

        emph_threshold = -950
        ct_list = os.listdir(self.in_ct_dir)

        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')
        os.makedirs(emph_mask_dir, exist_ok=True)

        def _process_single_case(ct_file_name):
            in_ct = os.path.join(self.in_ct_dir, ct_file_name)
            lung_mask = os.path.join(lung_mask_dir, ct_file_name)

            ct_img = nib.load(in_ct)
            lung_img = nib.load(lung_mask)

            ct_data = ct_img.get_fdata()
            lung_data = lung_img.get_fdata()

            emph_data = np.zeros(ct_data.shape, dtype=int)
            emph_data[(ct_data < emph_threshold) & (lung_data > 0)] = 1

            emph_img = nib.Nifti1Image(emph_data,
                                       affine=ct_img.affine,
                                       header=ct_img.header)
            emph_path = os.path.join(emph_mask_dir, ct_file_name)
            nib.save(emph_img, emph_path)

        Parallel(
            n_jobs=10,
            prefer='threads'
        )(delayed(_process_single_case)(ct_file_name)
          for ct_file_name in tqdm(ct_list, total=len(ct_list)))

    def get_emphysema_measurement(self):
        lung_mask_dir = os.path.join(self.project_dir, 'lung_mask')
        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')

        ct_file_list = os.listdir(lung_mask_dir)
        record_list = []
        for ct_file_name in ct_file_list:
            pid = ct_file_name.replace('.nii.gz', '')

            lung_mask = nib.load(os.path.join(lung_mask_dir, ct_file_name)).get_fdata()
            emph_mask = nib.load(os.path.join(emph_mask_dir, ct_file_name)).get_fdata()

            emph_score = 100. * np.count_nonzero(emph_mask) / np.count_nonzero(lung_mask)

            record_list.append({
                'pid': pid,
                'emph_score': emph_score
            })

        emph_score_df = pd.DataFrame(record_list)
        emph_score_csv = os.path.join(self.project_dir, 'emph.csv')
        print(f'Save to {emph_score_csv}')
        emph_score_df.to_csv(emph_score_csv, index=False)

# 1) Need the emphysema metrics on the inhalation and exhalation kernels (non harmonized)
# 2) Need the emphysema metric on the harmonizated inhalation scans (harmonized) 
# 3) Need the emphysema metric after resampling the images (non harmonized)
# 4) Need the emphysema metric after resampling the images (harmonized)
# 5) Maybe emphysema metric for harmonized and non harmonized registered images ? 

# non_harmonized = [("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/inspiratory_BONE", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/insp_BONE_emphysema"), #Inspiratory BONE non harmonized
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/expiratory_STANDARD", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/val_test/exp_BONE_emphysema"), #Expiratory STANDARD non harmonized 
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/cycleGAN_epoch5_results", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/insp_exp_run1_results_cycleGAN/cycleGAN_epoch5_results/harmonized_emphysema"), #Inspiratory BONE harmonized 
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_BONE", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/emphysema_resampled_BONE"), #Inspiratory BONE non harmonized resampled
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/resampled_masked_out_STANDARD", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/resampled/emphysema_resampled_STANDARD"), #Expiratory STANDARD non harmonized resampled
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled/emphysema_resampled_harmonized"), #Inspiratory BONE harmonized resampled
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/SyN_registered_STANDARD_expiratory_images", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/emphysema_SyN_registered_STANDARD_expiratory_images"), #Expiratory STANDARD non harmonized registered
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/SyN_registered_STANDARD_expiratory_images", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/emphysema_SyN_registered_STANDARD_expiratory_images")] #Expiratory STANDARD harmonized registered


# non_harmonized = [ #Expiratory STANDARD non harmonized 
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/resampled", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/emphysema_resampled_harmonized"), #Inspiratory BONE harmonized resampled
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/SyN_registered_STANDARD_expiratory_images", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/non_harmonized/emphysema_SyN_registered_STANDARD_expiratory_images"), #Expiratory STANDARD non harmonized registered
#                   ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/SyN_registered_STANDARD_expiratory_images", "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_data/harmonized/emphysema_SyN_registered_STANDARD_expiratory_images")] #Expiratory STANDARD harmonized registered



#Need in CT, lung mask and generate emphysema mask

inpath = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/emphysema_inhale_soft_kernel/NIfTI_file"
outpath = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/emphysema_inhale_soft_kernel"

print(f"Processing {inpath}")
print(f"Output directory: {outpath}")
os.makedirs(outpath, exist_ok=True)
emphysema_analysis = EmphysemaAnalysis(in_ct_dir = inpath, project_dir=outpath)
emphysema_analysis.generate_lung_mask()
emphysema_analysis.get_emphysema_mask()
emphysema_analysis.get_emphysema_measurement()