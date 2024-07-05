import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pydicom
import cv2 as cv
import nibabel as nib
from joblib import Parallel, delayed
import json


def load_json_config(config_file):
    f = open(config_file)
    config = json.load(f)
    return config


class ArchiveUtils:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _if_secondary_or_derived(ds):
        image_type = str(ds[0x08, 0x08].value) if (0x08, 0x08) in ds else None
        secondary_or_derived = ("DERIVED" in image_type) or (
                    "SECONDARY" in image_type) if image_type is not None else None
        return secondary_or_derived

    @staticmethod
    def _if_iv_contrast(ds):
        iv_contrast = False
        for item in ds:
            if ('Contrast' in item.keyword) or ('Bolus' in item.keyword):
                iv_contrast = True

        return iv_contrast

    @staticmethod
    def _get_modality(ds):
        return str(ds[0x08, 0x60].value) if (0x08, 0x60) in ds else None

    def archive_dcm(self):
        """
        1. Identify all dcm files under the input dir.
        2. Re-arrange the folder structure:
           + DICOM_DATA_ROOT
             + Series UID
               + DICOM images (*.dcm)
        :return:
        """
        in_dcm_dir = self.config['in_dcm_dir']
        out_dcm_dir = self.config['out_dcm_dir']
        os.makedirs(out_dcm_dir, exist_ok=True)

        print(f'Archive DICOM files')
        print(f'Input: {in_dcm_dir}')
        print(f'Output: {out_dcm_dir}')

        # Assume the first level folders are subjects (which usually is the case)
        subject_list = os.listdir(in_dcm_dir)
        print(f'Identified {len(subject_list)} subjects.')
        # for subject in tqdm(subject_list, total=len(subject_list)):
        for subject_index, subject in enumerate(subject_list):
            try:
                print(f'Process {subject} ({subject_index + 1} / {len(subject_list)})')
                # dcm_path_list = list(Path(os.path.join(in_dcm_dir, subject)).rglob('*.dcm'))
                # all_file_list = list(Path(os.path.join(in_dcm_dir, subject)).rglob('**/*'))
                dcm_path_list = []
                for dirpath, dirs, files in os.walk(os.path.join(in_dcm_dir, subject)):
                    for filename in files:
                        fname = os.path.join(dirpath, filename)
                        if os.path.isfile(fname):
                            dcm_path_list.append(fname)
                print(f'Identified {len(dcm_path_list)} dcm files.')
                # for dcm_path in Path(os.path.join(in_dcm_dir, subject)).rglob('*.dcm'):
                for dcm_path in tqdm(dcm_path_list, total=len(dcm_path_list)):
                    ds = pydicom.dcmread(dcm_path)

                    modality_str = self._get_modality(ds)
                    if modality_str != 'CT':
                        continue
                    if self._if_secondary_or_derived(ds) is not False:
                        continue

                    series_uid = str(ds[0x20, 0x0E].value)
                    out_dir = os.path.join(out_dcm_dir, series_uid)
                    os.makedirs(out_dir, exist_ok=True)

                    out_dcm_path = os.path.join(out_dir, os.path.basename(dcm_path))
                    ln_cmd = f'ln -sf {dcm_path} {out_dcm_path}'
                    os.system(ln_cmd)
            except:
                print(f'Failed to process {subject}.')

    def _convert_nii_single_series(self, series_dir, archive_dir):
        dcm_file_list = os.listdir(series_dir)
        if len(dcm_file_list) == 0:
            return
        first_dcm_path = os.path.join(series_dir, dcm_file_list[0])
        try:
            ds = pydicom.dcmread(first_dcm_path)
            if 'PixelData' in ds:
                delattr(ds, 'PixelData')
            series_uid = str(ds[0x20, 0x0e].value)
        except:
            print(f'Failed to read dcm data {first_dcm_path}')
            return {}

        included_tag_index_map = {
            'PatientID': (0x10, 0x20),
            'StudyDate': (0x08, 0x20),
            'Modality': (0x08, 0x60),
            'BodyPart': (0x18, 0x15),
            'Manufacturer': (0x08, 0x70),
            'SeriesUID': (0x20, 0x0E),
            'ConvolutionKernel': (0x18, 0x1210),
            'SliceThickness': (0x18, 0x50),
            'KVP': (0x18, 0x60),
            'Reconstruction Diameter': (0x18, 0x1100),
            'Exposure Time': (0x18, 0x1150),
            'X-Ray Tube Current': (0x18, 0x1151),
            'Exposure': (0x18, 0x1152),
            'Spiral Pitch Factor': (0x18, 0x9311)
        }

        image_info_dict = {}
        for tag, dcm_index in included_tag_index_map.items():
            image_info_dict[tag] = str(ds[dcm_index].value) if dcm_index in ds else None

        # 1. Convert to nii
        dcm2niix_path = self.config['dcm2niix']
        dcm2niix_cmd = f'{dcm2niix_path} -i y -s y -w 1 -z y -f %j ' \
                       f'-o {archive_dir} {series_dir}'
        print(dcm2niix_cmd)
        os.system(dcm2niix_cmd)

        # 2. Save header to json
        try:
            json_path = os.path.join(archive_dir, series_uid + '.json')
            header_str = ds.to_json()
            json_handle = open(json_path, 'w')
            json_handle.write(header_str)
            json_handle.close()
        except:
            print(f'Failed to generate the json file.')

        # 3. Generate combined png for 2D preview
        nii_path = os.path.join(archive_dir, series_uid + '.nii.gz')
        if os.path.exists(nii_path):
            try:
                out_png = os.path.join(archive_dir, series_uid + '.png')
                clip_obj = ClipMontagePlotNII(7, -150, 150)
                clip_obj.clip_plot(nii_path, out_png)
            except:
                print(f'Failed to generate png preview for {series_uid}')

        # 4. Get spacing and scan length from nii file
        if os.path.exists(nii_path):
            try:
                nii_obj = nib.load(nii_path)
                image_info_dict['Spacing'] = nii_obj.header['pixdim'][3]
                image_info_dict['NumSlice'] = nii_obj.header['dim'][3]
                image_info_dict['ScanLength'] = image_info_dict['Spacing'] * image_info_dict['NumSlice']
                image_info_dict['NumDICOM'] = len(dcm_file_list)
            except:
                print(f'Failed to read generate nii file for {series_uid}.')

        return image_info_dict

    def convert_nii(self):
        archived_dcm_dir = self.config['out_dcm_dir']
        out_nii_dir = self.config['out_nii_dir']
        os.makedirs(out_nii_dir, exist_ok=True)
        out_scan_index_csv = self.config['out_scan_index_csv']

        series_list = os.listdir(archived_dcm_dir)

        # scan_info_record_list = []
        # for series_uid in tqdm(series_list, total=len(series_list), desc='Convert dcm to nii'):
        #     scan_info_record = self._convert_nii_single_series(
        #         os.path.join(archived_dcm_dir, series_uid), out_nii_dir)
        #     scan_info_record_list.append(scan_info_record)

        scan_info_record_list = Parallel(
            n_jobs=self.config['nproc'],
            prefer="threads"
        )(delayed(self._convert_nii_single_series)(os.path.join(archived_dcm_dir, series_uid), out_nii_dir)
          for series_uid in tqdm(series_list, total=len(series_list), desc='Convert dcm to nii'))

        scan_info_record_df = pd.DataFrame(scan_info_record_list)
        print(f'Save to {out_scan_index_csv}')
        scan_info_record_df.to_csv(
            out_scan_index_csv, index=False,
            columns=[
                'PatientID',
                'StudyDate',
                'Modality',
                'BodyPart',
                'Manufacturer',
                'Spacing',
                'NumSlice',
                'NumDICOM',
                'ScanLength',
                'SeriesUID', 
                'ConvolutionKernel',
                'SliceThickness',
                'KVP',
                'Reconstruction Diameter',
                'Exposure Time',
                'X-Ray Tube Current',
                'Exposure',
                'Spiral Pitch Factor'
            ]
        )

    def add_inclusion_flag(self):
        scan_info_record_csv = self.config['out_scan_index_csv']
        print(f'Load {scan_info_record_csv}')
        scan_info_record_df = pd.read_csv(scan_info_record_csv)

        # Exclude missing slice
        scan_info_record_df = scan_info_record_df.loc[
            scan_info_record_df['NumSlice'] <= scan_info_record_df['NumDICOM']]

        # Exclude potential incomplete scan by 200mm thresshold
        scan_info_record_df = scan_info_record_df.loc[
            scan_info_record_df['ScanLength'] >= 200]

        include_series_list = []
        for modality_tag in ['ABDOMEN', 'CHEST']:
            modality_df = scan_info_record_df.loc[scan_info_record_df['BodyPart'] == modality_tag]
            for subject_id, subject_df in modality_df.groupby(by='PatientID'):
                for study_date, sess_df in subject_df.groupby(by='StudyDate'):
                    sess_df = sess_df.sort_values(by='Spacing', ignore_index=True)
                    include_series_list.append(sess_df.iloc[0]['SeriesUID'])

        if_include_list = []
        for index, record in scan_info_record_df.iterrows():
            series_uid = record['SeriesUID']
            if_include_list.append(int(series_uid in include_series_list))

        scan_info_record_df['if_include'] = if_include_list
        scan_info_record_df = scan_info_record_df.sort_values(by=['PatientID', 'StudyDate'])

        print(f'Update {scan_info_record_csv}')
        scan_info_record_df.to_csv(scan_info_record_csv, index=False)

    def generate_hierarchy_preview_png(self):
        scan_info_record_csv = self.config['out_scan_index_csv']
        print(f'Load {scan_info_record_csv}')
        scan_info_record_df = pd.read_csv(scan_info_record_csv)

        nii_archive_dir = self.config['out_nii_dir']
        preview_png_dir = self.config['out_preview_dir']
        os.makedirs(preview_png_dir, exist_ok=True)
        for index, scan_record in scan_info_record_df.iterrows():
            patient_id = str(scan_record['PatientID'])
            study_date = str(scan_record['StudyDate'])
            body_part = str(scan_record['BodyPart']) if scan_record['BodyPart'] is not np.nan else 'Unknown'
            series_uid = str(scan_record['SeriesUID'])

            if body_part is None:
                body_part = 'Unknown'
            in_png_path = os.path.join(nii_archive_dir, f'{series_uid}.png')
            if os.path.exists(in_png_path):
                out_dir = os.path.join(preview_png_dir, patient_id, study_date, body_part)
                os.makedirs(out_dir, exist_ok=True)
                out_png_path = os.path.join(out_dir, f'{series_uid}.png')
                cp_cmd = f'cp {in_png_path} {out_png_path}'
                os.system(cp_cmd)


class ClipMontagePlotNII:
    def __init__(
            self,
            num_clip,
            vmin, vmax
    ):
        self._num_clip = num_clip
        self._vmin = vmin
        self._vmax = vmax

    def clip_plot(
            self,
            in_nii,
            out_png
    ):
        montage_image = self._get_concatenated_nii_montage(in_nii)

        print(f'Output montage image to {out_png}')
        cv.imwrite(out_png, montage_image)

    def clip_plot_combine_cxr(
            self,
            in_nii,
            in_cxr,
            out_png
    ):
        ct_montage_image = self._get_concatenated_nii_montage(in_nii)

        dim_size = ct_montage_image.shape[0]
        print(f'Load {in_cxr}')
        cxr_image = cv.imread(in_cxr, cv.IMREAD_UNCHANGED)
        cxr_image = cv.resize(cxr_image, dsize=(dim_size, dim_size), interpolation=cv.INTER_CUBIC)

        concate_image = np.concatenate([ct_montage_image, cxr_image], axis=1)
        print(f'Write png to {out_png}')
        cv.imwrite(out_png, concate_image)

    def _get_concatenated_nii_montage(
            self,
            in_nii
    ):
        print(f'Load {in_nii}')
        image_obj = nib.load(in_nii)
        in_data = image_obj.get_data()
        if len(in_data.shape) == 4:
            in_data = in_data[:, :, :, 0]

        pixdim = image_obj.header['pixdim'][1:4]
        print(in_data.shape)

        dim_x, dim_y, dim_z = np.multiply(np.array(in_data.shape), pixdim).astype(int)

        # pixdim_xy = image_obj.header['pixdim'][2]
        # z_scale_ratio = pixdim_z / pixdim_xy
        # print(f'z_scale_ratio: {z_scale_ratio:.2f}')
        print(f'Input dimensions:')
        print(in_data.shape)
        # z_dim = int(z_scale_ratio * in_data.shape[2])
        # xy_dim = in_data.shape[0]

        # dim_vector = [in_data.shape[0], in_data.shape[1], z_dim]
        dim_vector = [dim_x, dim_y, dim_z]
        print(f'After normalization')
        print(dim_vector)
        max_dim = np.max(np.array(dim_vector))

        # Step.1 Get all clip.
        # Step.2 Pad to the same size (cv2)
        # Step.3 Concatenate into montage view

        view_flag_list = ['sagittal', 'coronal', 'axial']
        view_image_list = []
        for idx_view in range(len(view_flag_list)):
            view_flag = view_flag_list[idx_view]
            clip_list = []
            for idx_clip in range(self._num_clip):
                clip = self._clip_image(in_data, view_flag, self._num_clip, idx_clip)
                clip = self._rescale_to_0_255(clip, self._vmin, self._vmax)
                # if (view_flag == 'sagittal') | (view_flag == 'coronal'):
                #     clip = cv.resize(clip, (xy_dim, z_dim), interpolation=cv.INTER_CUBIC)
                if view_flag == 'sagittal':
                    clip = cv.resize(clip, (dim_y, dim_z), interpolation=cv.INTER_CUBIC)
                elif view_flag == 'coronal':
                    clip = cv.resize(clip, (dim_x, dim_z), interpolation=cv.INTER_CUBIC)
                elif view_flag == 'axial':
                    clip = cv.resize(clip, (dim_x, dim_y), interpolation=cv.INTER_CUBIC)
                clip = self._pad_to(clip, max_dim, max_dim)
                clip = np.clip(clip, 0, 255)
                clip = np.uint8(clip)
                clip_list.append(clip)
            view_image = np.concatenate(clip_list, axis=1)
            view_image_list.append(view_image)

        montage_image = np.concatenate(view_image_list, axis=0)
        montage_image = cv.resize(montage_image, dsize=(self._num_clip * 512, 3 * 512),
                                  interpolation=cv.INTER_NEAREST)

        return montage_image

    @staticmethod
    def _pad_to(in_clip, pad_dim_x, pad_dim_y):
        dim_x, dim_y = in_clip.shape[0], in_clip.shape[1]
        dim_x_pad_before = int((pad_dim_x - dim_x) / 2)
        dim_x_pad_after = (pad_dim_x - dim_x) - dim_x_pad_before
        dim_y_pad_before = int((pad_dim_y - dim_y) / 2)
        dim_y_pad_after = (pad_dim_y - dim_y) - dim_y_pad_before
        out_pad = np.pad(in_clip, ((dim_x_pad_before, dim_x_pad_after),
                                   (dim_y_pad_before, dim_y_pad_after)),
                         constant_values=0)
        return out_pad

    @staticmethod
    def _rescale_to_0_255(in_img_data, vmin, vmax):
        img_data = np.clip(in_img_data, vmin, vmax)
        cv.normalize(img_data, img_data, 0, 255, cv.NORM_MINMAX)

        return img_data

    @staticmethod
    def _clip_image(image_data, clip_plane, num_clip=1, idx_clip=0):
        im_shape = image_data.shape

        # Get clip offset
        idx_dim = -1
        if clip_plane == 'sagittal':
            idx_dim = 0
        elif clip_plane == 'coronal':
            idx_dim = 1
        elif clip_plane == 'axial':
            idx_dim = 2
        else:
            raise NotImplementedError

        clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
        offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

        clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

        clip = None
        if clip_plane == 'sagittal':
            clip = image_data[clip_location, :, :]
            clip = np.flip(clip, 0)
            clip = np.rot90(clip)
        elif clip_plane == 'coronal':
            clip = image_data[:, clip_location, :]
            clip = np.rot90(clip)
        elif clip_plane == 'axial':
            clip = image_data[:, :, clip_location]
            clip = np.rot90(clip)
        else:
            raise NotImplementedError

        return clip


class AppDataGenerator:
    def __init__(self, config):
        self.config = config
        self.app_root = self.config['out_app_dir']
        os.makedirs(self.app_root, exist_ok=True)
        self.dcm_dir = self.config['out_dcm_dir']
        self.nii_dir = self.config['out_nii_dir']
        self.scan_df = pd.read_csv(self.config['out_scan_index_csv'])
        self.scan_df = self.scan_df.loc[self.scan_df['if_include'] == 1]

    def generate_app_data(self, app_tag):
        if app_tag == 'liver_ai':
            self.generate_liver_ai_data()
        elif app_tag == 'lung_body_composition':
            self.generate_lung_body_composition_data()
        elif app_tag == 'abdomen_muscle_fat':
            self.generate_abdomen_muscle_fat_data()
        elif app_tag == 'abdomen_body_composition':
            self.generate_abdomen_body_composition_data()

    def generate_liver_ai_data(self):
        print(f'Create Liver AI input data')

        project_data_dir = os.path.join(self.app_root, 'liver_AI')
        os.makedirs(project_data_dir, exist_ok=True)

        dcm_dir = os.path.join(project_data_dir, 'Input')
        os.makedirs(dcm_dir, exist_ok=True)

        cohort_df = self.scan_df.loc[(self.scan_df['BodyPart'] == 'ABDOMEN') & (self.scan_df['if_include'] == 1)]
        print(f'Identified {len(cohort_df.index)} scans.')
        for index, scan_record in tqdm(cohort_df.iterrows(), total=len(cohort_df.index)):
            patient_id = str(scan_record['PatientID'])
            study_date = str(scan_record['StudyDate'])
            series_uid = str(scan_record['SeriesUID'])

            out_scan_dcm_dir = os.path.join(dcm_dir, patient_id, study_date, series_uid)
            os.makedirs(out_scan_dcm_dir, exist_ok=True)

            in_dcm_dir = os.path.join(self.dcm_dir, series_uid)
            rsync_cmd = f'rsync -L -hav -q --progress {in_dcm_dir}/ {out_scan_dcm_dir}/'
            os.system(rsync_cmd)

    def generate_lung_body_composition_data(self):
        print(f'Create input data for chest CT-based body composition analysis')

        project_data_dir = os.path.join(self.app_root, 'lung_body_composition')
        os.makedirs(project_data_dir, exist_ok=True)

        project_input_data_dir = os.path.join(project_data_dir, 'Input')
        out_nii_dir = os.path.join(project_input_data_dir, 'NIFTI')
        os.makedirs(out_nii_dir, exist_ok=True)

        cohort_df = self.scan_df.loc[(self.scan_df['BodyPart'] == 'CHEST') & (self.scan_df['if_include'] == 1)]
        print(f'Identified {len(cohort_df.index)} scans.')

        in_nii_dir = self.config['out_nii_dir']
        for index, scan_record in tqdm(cohort_df.iterrows(), total=len(cohort_df.index)):
            patient_id = str(scan_record['PatientID'])
            study_date = str(scan_record['StudyDate'])
            series_uid = str(scan_record['SeriesUID'])

            in_nii = os.path.join(in_nii_dir, f'{series_uid}.nii.gz')
            out_nii = os.path.join(out_nii_dir, f'{patient_id}_{study_date}.nii.gz')
            cp_cmd = f'cp {in_nii} {out_nii}'
            os.system(cp_cmd)

    def _generate_abdomen_nii_input(self, project_input_data_dir):
        out_nii_dir = os.path.join(project_input_data_dir, 'NIFTI')
        os.makedirs(out_nii_dir, exist_ok=True)

        cohort_df = self.scan_df.loc[(self.scan_df['BodyPart'] == 'ABDOMEN') & (self.scan_df['if_include'] == 1)]
        print(f'Identified {len(cohort_df.index)} scans.')

        in_nii_dir = self.config['out_nii_dir']
        for index, scan_record in tqdm(cohort_df.iterrows(), total=len(cohort_df.index)):
            patient_id = str(scan_record['PatientID'])
            study_date = str(scan_record['StudyDate'])
            series_uid = str(scan_record['SeriesUID'])

            in_nii = os.path.join(in_nii_dir, f'{series_uid}.nii.gz')
            out_nii = os.path.join(out_nii_dir, f'{patient_id}_{study_date}.nii.gz')
            cp_cmd = f'cp {in_nii} {out_nii}'
            os.system(cp_cmd)

    def generate_abdomen_muscle_fat_data(self):
        print(f'Create input data for abdomen muscle and fat analysis (2D)')

        project_data_dir = os.path.join(self.app_root, 'abdomen_muscle_fat')
        os.makedirs(project_data_dir, exist_ok=True)

        self._generate_abdomen_nii_input(project_data_dir)

    def generate_abdomen_body_composition_data(self):
        print(f'Create input data for abdomen body composition analysis')

        project_data_dir = os.path.join(self.app_root, 'abdomen_body_composition')
        os.makedirs(project_data_dir, exist_ok=True)

        self._generate_abdomen_nii_input(project_data_dir)
