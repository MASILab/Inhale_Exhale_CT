import os 
import pandas as pd 
from tqdm import tqdm

dataframe = pd.read_csv("/fs5/p_masi/krishar1/COPDGENE/COPDgene_SPIE2025.csv")

#Get the list of subjects
subjects = dataframe['File Name'].tolist()

source = "/fs5/p_masi/krishar1/COPDGENE/SPIE_2025"
target = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/raw_data"

print(subjects)

#Symlink the data from source to target
for subject in tqdm(subjects):
    if subject in os.listdir(source):
        os.symlink(os.path.join(source, subject), os.path.join(target, subject))