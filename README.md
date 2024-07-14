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


