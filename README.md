# Inhale_Exhale_CT
Repository for harmonization of reconstruction kernels on inspiratory, expiratory scans


## Data Prep: 
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


### Creating the COPDGENE curated dataset from the spreadsheet 
1) **COPDGene_characterization.ipynb**: Looked at the data. Downloaded the subjects and created spreadsheets for downloaded subjects. Cleaned the dataset by changing the labels of GOLD classification to have standard values across them. Saved the spreadsheet at /fs5/p_masi/krishar1/COPDGENE/COPD_gene_7368subjects.csv

2) **datasplot_copdgene.ipynb**: Loaded the saved spreadhseet from the previous python notebook. Cleaned the dataset again to only account for GOLD stages 2-4. Dropped everything else. saved spreadhseet to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/COPD_gene_COPDsubjects.csv
    - Created a file **prepCOPDgenedataSPIE.py**
    - Used the function **curate_copd_data_SPIE()** function to read through the dicom tags in the image folder of COPD and return a spreadsheet with all the folders that have inspiratory and expiratory scans having dicom images based on the GOLD criteria. Save spreadsheet to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_goldcriteria_SPIE_data.csv
    -  Go back to the python notebook and explore convolution kernels. Drop all the kernels that are not from GE manufacturer. Save to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_goldcriteria_SPIE_data_GEkernels.csv"
    - Got back to the python script and run the function **get_inspiratory_expiratory_GEkernels()** to create a column that tells us if a scan is either inspiratory or expiratory. Save to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/copdgene_GOLD_GEkernels_matched.csv
    - In the jupyter notebook, do the analysis and randomly sampel 100 subjects. Check if paired scans exist for the subjects. Symlink data and run the docker code to convert DICOM to NIfTI. 

3) **sampledmoreGEkernels_SPIE.ipyng**: Ater conversion, use the spreadsheet and remove the failed QA subjects that only produced subjects with a single kernel. Save to /nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/final_paired_data_spreadsheets/COPD_GE_pairedkernels_initial_sampling.csv
    - Get addiitonal subjects since 100 subjects were not obtained. Once attained, synlink the data. Run the DICOM to NIfTI conversion and obtain the spreadsheet
    - Repeat the data filtering process and get a new spreadsheet with paired kernels 

4) **train_test_split.ipynb**: Merge dataframes of initial sampling and additional sampling. Add splits of train, val and test for the purpose of training cycleGAN.

