import pandas as pd 
import matplotlib.pyplot as plt
import os 
import shutil 


copd_df = pd.read_csv("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/final_paired_data_spreadsheets/COPD_GE_pairedkernels_100randomsamples_GOLDlabels.csv")
#Drop duplicates in the dataframe column SeriesDescription 
copd_df = copd_df.drop_duplicates(subset = "SeriesDescription")
print(copd_df)