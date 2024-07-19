import pandas as pd 
import matplotlib.pyplot as plt
import os 
import shutil 
from tqdm import tqdm

def drop_duplicates():
    copd_df = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/final_paired_data_spreadsheets/COPD_GE_pairedkernels_100randomsamples_GOLDlabels.csv")
    #Drop duplicates in the dataframe column SeriesDescription 
    copd_df = copd_df.drop_duplicates(subset = "SeriesDescription")
    copd_df.reset_index(drop = True, inplace = True)
    copd_df.to_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/final_paired_data_spreadsheets/COPD_GE_pairedkernels_100randomsamples_GOLDlabels.csv", index = False)
    print(copd_df)

# drop_duplicates()


def read_train_test_split_copy_data_controls():
    #location for data splits: /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split
    #Location for data_samples: /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/dcm2nii/NIfTI
    #Location for additional_data: /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/dcm2nii_additional_sample/NIfTI
    #Location for controls: /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/dcm2nii_controls/NIfTI

    controls = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/COPD_controls_SPIE.csv")
    controls_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/dcm2nii_controls/NIfTI"

    output_train_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train/inspiratory_BONE"
    output_train_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train/expiratory_STANDARD" 

    output_val_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/inspiratory_BONE"
    output_val_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/expiratory_STANDARD"

    output_test_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/inspiratory_BONE"
    output_test_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/expiratory_STANDARD"

    n_train = 0
    n_val = 0 
    n_test = 0
    
    #controls
    for index, row in tqdm(controls.iterrows()):
        if row["split"] == "train":
            pid = row['PatientID']
            bone = row['SeriesUID_BONE'] + ".nii.gz"
            std = row['SeriesUID_STANDARD'] + ".nii.gz"

            #I want to copy the file and renmae the file to the patient ID + the kernel 
            os.makedirs(output_train_bone, exist_ok = True)
            os.makedirs(output_train_std, exist_ok = True)
            shutil.copy(os.path.join(controls_path, bone), os.path.join(output_train_bone, pid + "_BONE_control.nii.gz"))
            shutil.copy(os.path.join(controls_path, std), os.path.join(output_train_std, pid + "_STANDARD_control.nii.gz"))
            n_train += 1

            if n_train == 25:
                break
    
    for index, row in tqdm(controls.iterrows()):
        if row["split"] == "val":
            pid = row['PatientID']
            bone = row['SeriesUID_BONE'] + ".nii.gz"
            std = row['SeriesUID_STANDARD'] + ".nii.gz"

            #I want to copy the file and renmae the file to the patient ID + the kernel 
            os.makedirs(output_val_bone, exist_ok = True)
            os.makedirs(output_val_std, exist_ok = True)
            shutil.copy(os.path.join(controls_path, bone), os.path.join(output_val_bone, pid + "_BONE_control.nii.gz"))
            shutil.copy(os.path.join(controls_path, std), os.path.join(output_val_std, pid + "_STANDARD_control.nii.gz"))
            n_val += 1

            if n_val == 10:
                break
    
    for index, row in tqdm(controls.iterrows()):
        if row["split"] == "test":
            pid = row['PatientID']
            bone = row['SeriesUID_BONE'] + ".nii.gz"
            std = row['SeriesUID_STANDARD'] + ".nii.gz"

            #I want to copy the file and renmae the file to the patient ID + the kernel 
            os.makedirs(output_test_bone, exist_ok = True)
            os.makedirs(output_test_std, exist_ok = True)
            shutil.copy(os.path.join(controls_path, bone), os.path.join(output_test_bone, pid + "_BONE_control.nii.gz"))
            shutil.copy(os.path.join(controls_path, std), os.path.join(output_test_std, pid + "_STANDARD_control.nii.gz"))
            n_test += 1

            if n_test == 10:
                break


def train_test_split_copy_cases():
    cases_100 = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/COPD_cases_GE_SPIE.csv")

    #choose 15 random controls and 15 random cases based on the split == "train"

    cases_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/dcm2nii/NIfTI"
    cases_path_additional = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/dcm2nii_additional_sample/NIfTI" 

    output_train_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train/inspiratory_BONE"
    output_train_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/train/expiratory_STANDARD"

    output_val_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/inspiratory_BONE"
    output_val_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/expiratory_STANDARD"

    output_test_bone = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/inspiratory_BONE"
    output_test_std = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/test/expiratory_STANDARD"

    exclude_ids = ['1.2.840.113619.2.108.3994720393.31775.1243423826.241', '1.2.840.113619.2.108.3994720393.31775.1243423829.349']

    

    for i, row in cases_100.iterrows():
        if row["SeriesUID_BONE"] == "1.2.840.113619.2.108.3994720393.31775.1243423826.241" and row["SeriesUID_STANDARD"] == "1.2.840.113619.2.108.3994720393.31775.1243423829.349":
            print("Found the subject! Dropping since image does not exist")
            cases_100.drop(i, inplace = True)
        else:
            print("Error: UIDS not found!")
    

    filtered_df_gold2 = cases_100[(cases_100['split'] == 'train') & (cases_100['GOLD'] == 'GOLD_2')].head(15)
    filtered_df_gold3 = cases_100[(cases_100['split'] == 'train') & (cases_100['GOLD'] == 'GOLD_3')].head(10)

    filtered_train_df = pd.concat([filtered_df_gold2, filtered_df_gold3])
    print(filtered_train_df)

    filtered_df_val_gold2 = cases_100[(cases_100['split'] == 'val') & (cases_100['GOLD'] == 'GOLD_2')].head(10)
    filtered_df_val_gold3 = cases_100[(cases_100['split'] == 'val') & (cases_100['GOLD'] == 'GOLD_3')].head(5)

    filtered_val_df = pd.concat([filtered_df_val_gold2, filtered_df_val_gold3])
    print(filtered_val_df)

    filtered_df_test_gold2 = cases_100[(cases_100['split'] == 'test') & (cases_100['GOLD'] == 'GOLD_2')].head(10)
    filtered_df_test_gold3 = cases_100[(cases_100['split'] == 'test') & (cases_100['GOLD'] == 'GOLD_3')].head(5)

                                                                                                              
    filtered_test_df = pd.concat([filtered_df_test_gold2, filtered_df_test_gold3])
    print(filtered_test_df)


    for index, row in tqdm(filtered_train_df.iterrows()):
        pid = row['PatientID']
        bone = row['SeriesUID_BONE'] + ".nii.gz"
        std = row['SeriesUID_STANDARD'] + ".nii.gz"

        os.makedirs(output_train_bone, exist_ok = True)
        os.makedirs(output_train_std, exist_ok = True)
        if bone in os.listdir(cases_path) and std in os.listdir(cases_path):
            shutil.copy(os.path.join(cases_path, bone), os.path.join(output_train_bone, pid + "_BONE.nii.gz"))
            shutil.copy(os.path.join(cases_path, std), os.path.join(output_train_std, pid + "_STANDARD.nii.gz"))
        else:
            shutil.copy(os.path.join(cases_path_additional, bone), os.path.join(output_train_bone, pid + "_BONE.nii.gz"))
            shutil.copy(os.path.join(cases_path_additional, std), os.path.join(output_train_std, pid + "_STANDARD.nii.gz"))
    
    for index, row in tqdm(filtered_val_df.iterrows()):
        pid = row['PatientID']
        bone = row['SeriesUID_BONE'] + ".nii.gz"
        std = row['SeriesUID_STANDARD'] + ".nii.gz"

        os.makedirs(output_val_bone, exist_ok = True)
        os.makedirs(output_val_std, exist_ok = True)
        if bone in os.listdir(cases_path) and std in os.listdir(cases_path):
            shutil.copy(os.path.join(cases_path, bone), os.path.join(output_val_bone, pid + "_BONE.nii.gz"))
            shutil.copy(os.path.join(cases_path, std), os.path.join(output_val_std, pid + "_STANDARD.nii.gz"))
        else:
            shutil.copy(os.path.join(cases_path_additional, bone), os.path.join(output_val_bone, pid + "_BONE.nii.gz"))
            shutil.copy(os.path.join(cases_path_additional, std), os.path.join(output_val_std, pid + "_STANDARD.nii.gz"))


    for index, row in tqdm(filtered_test_df.iterrows()):
        pid = row['PatientID']
        bone = row['SeriesUID_BONE'] + ".nii.gz"
        std = row['SeriesUID_STANDARD'] + ".nii.gz"

        os.makedirs(output_test_bone, exist_ok = True)
        os.makedirs(output_test_std, exist_ok = True)

        if bone in os.listdir(cases_path) and std in os.listdir(cases_path):
            shutil.copy(os.path.join(cases_path, bone), os.path.join(output_test_bone, pid + "_BONE.nii.gz"))
            shutil.copy(os.path.join(cases_path, std), os.path.join(output_test_std, pid + "_STANDARD.nii.gz"))
        else:
            shutil.copy(os.path.join(cases_path_additional, bone), os.path.join(output_test_bone, pid + "_BONE.nii.gz"))
            shutil.copy(os.path.join(cases_path_additional, std), os.path.join(output_test_std, pid + "_STANDARD.nii.gz"))


read_train_test_split_copy_data_controls()

# train_test_split_copy_cases()