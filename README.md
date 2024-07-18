# Inhale_Exhale_CT
Repository for harmonization of reconstruction kernels on inspiratory, expiratory scans

### Creating the COPDGENE curated dataset from the spreadsheet (Needs to be done in a single file in the future)
### Cycle GAN data curation
1) **COPDGene_characterization.ipynb**: Looked at the data. Downloaded the subjects and created spreadsheets for downloaded subjects. Cleaned the dataset by changing the labels of GOLD classification to have standard values across them. Saved the spreadsheet at /fs5/p_masi/krishar1/COPDGENE/COPD_gene_7368subjects.csv

2) **datasplot_copdgene.ipynb**: Loaded the saved spreadhseet from the previous python notebook. Cleaned the dataset again to only account for GOLD stages 2-4. Dropped everything else. saved spreadhseet to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/COPD_gene_COPDsubjects.csv
    - Created a file **prepCOPDgenedataSPIE.py**
    - Used the function **curate_copd_data_SPIE()** function to read through the dicom tags in the image folder of COPD and return a spreadsheet with all the folders that have inspiratory and expiratory scans having dicom images based on the GOLD criteria. Save spreadsheet to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_goldcriteria_SPIE_data.csv
    -  Go back to the python notebook and explore convolution kernels. Drop all the kernels that are not from GE manufacturer. Save to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_goldcriteria_SPIE_data_GEkernels.csv"
    - Got back to the python script and run the function **get_inspiratory_expiratory_GEkernels()** to create a column that tells us if a scan is either inspiratory or expiratory. Save to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_GOLD_GEkernels_matched.csv
    - In the jupyter notebook, do the analysis and randomly sampel 100 subjects. Check if paired scans exist for the subjects. Symlink data and run the docker code to convert DICOM to NIfTI. 

3) **sampledmoreGEkernels_SPIE.ipynb**: Ater conversion, use the spreadsheet and remove the failed QA subjects that only produced subjects with a single kernel. Save to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/final_paired_data_spreadsheets/COPD_GE_pairedkernels_initial_sampling.csv
    - Get addiitonal subjects since 100 subjects were not obtained. Once attained, synlink the data. Run the DICOM to NIfTI conversion and obtain the spreadsheet
    - Repeat the data filtering process and get a new spreadsheet with paired kernels 

4) **classifyGOLDfor100sampledCOPD.ipynb**: GOLD classification labels for the existing sampled data. Merge the 100 sampled dataframe and the additional sample dataframes to get the final dataframe. 
    - There were 134 subjects. We split these into two datframes, one with 100 randomly subjects and another with the remaining 34 subjects
    - For the 100 subjects, the final dataframe had duplicates which need to be removed.
    - The spreadsheet needs to be corrected because duplicates exists. 
    - Some values were repeated again and the dataframe size went from 200 to 248. However, labels are correct.

5) **samplecontrols.ipynb**: Sampling all controls. Contains entire code workflow from the first 4 steps in one notebook (Easy for future reference)

6) **train_test_split.py**: Splitting the labels for cases and controls to finalize the data for training the model.
    - Drop duplicates in the file /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/final_paired_data_spreadsheets/COPD_GE_pairedkernels_100randomsamples_GOLDlabels.csv

7) **traintestsplit_dev.ipynb**: Script to do the train, test split for controls and cases. Very easy to follows


- Voxel morph needs its own data curation (To do)



### Data Prep: 
Step 1: Have all the data under a single directory. Ensure that only DICOM files exist in the subdirectories 

Step 2: Create a config file with the following parameters: 
    "dcm2niix": Location of dcm2niix tool,
    "nproc": Number of threads
    "in_dcm_dir": Input direcotry with all the DICOM files
    "out_dcm_dir": Archive file to output DICOM directories
    "out_nii_dir": Output direcotry for the NIfTIs
    "out_preview_dir": Output png folder with triplanar screenshots
    "out_scan_index_csv": Spreadsheet with extracted DICOM information from the DICOM tags
    "out_app_dir": Directory for app data 

Step 3: Configure dicomtonii.sh under the make_data folder and run the bash script