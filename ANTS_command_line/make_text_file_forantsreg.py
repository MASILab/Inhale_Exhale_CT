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
# with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harmonized.txt", "w") as f:
#     for fixed, moving in zip(harmonized_fixed, moving_one_files):
#         f.write(f"{harmonized_fixed_path}{fixed} {moving_path}{moving}\n")

#Create chunk of 10 for harmonized and non harmonized images 
#first ten images will be done on metallica and masi-34. Do the next 10 on masi-59 and masi-57 followed by the A6000! 

chunk1_fixed = fixed_one_files[10:20]
chunk1_moving = moving_one_files[10:20]
chunk1_harmonized = harmonized_fixed[10:20]

chunk2_fixed = fixed_one_files[20:30]
chunk2_moving = moving_one_files[20:30]
chunk2_harmonized = harmonized_fixed[20:30]

# with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/non_harmonized_chunk1.txt", "w") as f:
#     for fixed, moving in zip(chunk1_fixed, chunk1_moving):
#         f.write(f"{fixed_path}{fixed} {moving_path}{moving}\n")

# with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harmonized_chunk1.txt", "w") as f:
#     for fixed, moving in zip(chunk1_harmonized, chunk1_moving):
#         f.write(f"{harmonized_fixed_path}{fixed} {moving_path}{moving}\n")

a6000_path_fixed = "/local_ssd1/krishar1/registration_ANTS_command_line/images/clipped_masked_out_BONE/"
a6000_path_standard = "/local_ssd1/krishar1/registration_ANTS_command_line/images/clipped_masked_out_STANDARD/"
a6000_path_harmonized = "/local_ssd1/krishar1/registration_ANTS_command_line/images/clipped_masked_out_harmonized/"

with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_chunking_scripts/non_harmonized_chunk2.txt", "w") as f:
    for fixed, moving in zip(chunk2_fixed, chunk2_moving):
        f.write(f"{a6000_path_fixed}{fixed} {a6000_path_standard}{moving}\n")

with open("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/ANTS_chunking_scripts/harmonized_chunk2.txt", "w") as f:
    for fixed, moving in zip(chunk2_harmonized, chunk2_moving):
        f.write(f"{a6000_path_harmonized}{fixed} {a6000_path_standard}{moving}\n")
