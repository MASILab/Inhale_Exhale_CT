import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil 
from tqdm import tqdm
import pydicom as pyd

def curate_copd_data_SPIE():
    copd = pd.read_csv('/fs5/p_masi/krishar1/COPDGENE/COPD_gene_COPDsubjects.csv')
    file_names = copd['File Name'].to_list()
    source = '/fs5/p_masi/krishar1/COPDGENE/SPIE_2025'

    #Append all files to a list 
    dcmfilesfinal = []

    for file in tqdm(file_names):
        if file in os.listdir(source):
            dirpath = os.path.join(source, file)
            subdirs = os.listdir(dirpath)
            if len(subdirs) == 1:
                sub_sudir = subdirs[0]
                if sub_sudir == "19000101":
                    sub_sudir_path = os.path.join(dirpath, sub_sudir)
                    files = os.listdir(sub_sudir_path)
                    for subfile in files:
                        dmcfilepath = os.path.join(sub_sudir_path, subfile)
                        dcmfilesfinal.append(dmcfilepath)
                else:
                    sub_sudir_path = os.path.join(dirpath, sub_sudir)
                    dcmfilesfinal.append(sub_sudir_path)
            else:
                for subdir in subdirs:
                    sub_sudir_path = os.path.join(dirpath, subdir)
                    dcmfilesfinal.append(sub_sudir_path)

    # print(dcmfilesfinal)
                

    #Append all the files as a column in a new dataframe 
    copd_files = {'File_Path': dcmfilesfinal}
    df = pd.DataFrame(copd_files)

    df.to_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_goldcriteria_SPIE_data_debug.csv", index=False)

    #Loop through all the files and extract the following meta data: Reconstruction kernel, series description, manufacturer
    for index, row in tqdm(df.iterrows()):
        dcm = pyd.dcmread(os.path.join(row['File_Path'],os.listdir(row['File_Path'])[0])) #Read the first dicom file
    
        # Check if specific DICOM tags are present
        tags = ['PatientID','ConvolutionKernel', 'SeriesDescription', 'Manufacturer']
        for tag in tags:
            if tag in dir(dcm):
                df.at[index, tag] = getattr(dcm, tag)
            else:
                df.at[index, tag] = pd.NA

    print(df)
    df.to_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_goldcriteria_SPIE_data.csv", index=False)


curate_copd_data_SPIE()