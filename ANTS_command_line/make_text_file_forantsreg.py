import os 


fixed_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/images/clipped_masked_out_BONE/"
moving_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/images/clipped_masked_out_STANDARD/"
harmonized_fixed_path = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/images/clipped_masked_out_harmonized/"

fixed_one_files = sorted(os.listdir(fixed_path))
moving_one_files = sorted(os.listdir(moving_path))

harmonized_fixed = sorted(os.listdir(harmonized_fixed_path))

#Make two txt files, one for harmonized and another for non harmonized 
#First, make the text file for the non harmonized images

# with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/non_harmonized.txt", "w") as f:
#     for fixed, moving in zip(fixed_one_files, moving_one_files):
#         f.write(f"{fixed_path}{fixed} {moving_path}{moving}\n")

#Now, make the text file for the harmonized images
with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harmonized.txt", "w") as f:
    for fixed, moving in zip(harmonized_fixed, moving_one_files):
        f.write(f"{harmonized_fixed_path}{fixed} {moving_path}{moving}\n")

