import pandas as pd
from Emphysemamodel.lungmask import ProcessLungMask
import logging
import os
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil


logger = logging.getLogger()

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Set GPU ID to 1


class EmphysemaAnalysis:
    def __init__(self, in_ct_dir, project_dir, lung_mask_dir):
        self.in_ct_dir = in_ct_dir
        self.project_dir = project_dir
        self.lung_mask_dir = lung_mask_dir

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
        lung_mask_dir = os.path.join(self.lung_mask_dir)

        emph_threshold = -950

        ct_list = os.listdir(self.in_ct_dir)

        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')
        os.makedirs(emph_mask_dir, exist_ok=True)

        def _process_single_case(ct_file_name):
            in_ct = os.path.join(self.in_ct_dir, ct_file_name)
            lung_mask = os.path.join(lung_mask_dir,ct_file_name)

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
        lung_mask_dir = os.path.join(self.lung_mask_dir)
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


reg_non_harm = [("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exptoinsp_harmonized"), 
              ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized_emphysema/lung_masks"),
              ("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized_emphysema")]
paths = reg_non_harm[0]


# for file in tqdm(os.listdir(paths)):
#     files = os.listdir(os.path.join(paths, file))


#     for image in files:
#         if image.endswith("_Warped.nii.gz"):
#             out_file = os.path.join(paths, file, image)
#             print(out_file)
    
#     #Copy the output file to the correct directory
#     shutil.copy(out_file, "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exptoinsp_harmonized_emphysema/images")


outpath ="/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized_emphysema"
inpath = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized_emphysema/images"
lung_mask = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_outputs_exp_toinsp_nonharmonized_emphysema/lung_masks"

print(f"Processing {inpath}")
print(f"Output directory: {outpath}")
os.makedirs(outpath, exist_ok=True) 

emphysema_analysis = EmphysemaAnalysis(in_ct_dir = inpath, project_dir=outpath, lung_mask_dir=lung_mask)
# emphysema_analysis.generate_lung_mask()
emphysema_analysis.get_emphysema_mask()
emphysema_analysis.get_emphysema_measurement()