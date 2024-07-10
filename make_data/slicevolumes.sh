 #!/bin/bash
 
 # Source directory containing the original NIfTI files
 source_dir=""
 
 # Target directory to store the output slices
 target_dir=""
 
 # Create the target directory if it doesn't exist
 mkdir -p "$target_dir"
 
 # Loop through each NIfTI file in the source directory
 for file in "$source_dir"/*.nii.gz; do
     # Extract the subject ID from the file name (assuming the file name format is "subjectID.nii.gz")
     subject_id=$(basename "$file" .nii.gz)
 
     # Use c3d to slice the NIfTI file and save the slices with subject ID and slice number as the file name
     c3d "$file" -slice z 0%:100% -oo "$target_dir/${subject_id}_%03d.nii.gz"
 done
