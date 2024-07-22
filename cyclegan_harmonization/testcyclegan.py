import torch 
import os 
from glob import glob 
from tqdm import tqdm 
from test_dataloader_custom import InferenceDataloader
import numpy as np
import nibabel as nib 
from torch.utils.data import DataLoader, Dataset 
from models.networks import define_G
from collections import OrderedDict 




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_dict = torch.load("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments/inspiration_expiration_COPD/inspiration_expiration_COPD/15_net_G_A.pth")
ord_dict = OrderedDict()
for k,v in gen_dict.items():
    ord_dict["module." + k] = v


generator = define_G(input_nc=1, output_nc=1, ngf=64, netG="resnet_9blocks", norm="instance", use_dropout=False, init_type="normal", init_gain=0.02, gpu_ids=[0])
generator.load_state_dict(ord_dict)

in_nii_path = glob(os.path.join("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/valid/inspiratory_BONE", '*.nii.gz'))
out_nii = os.path.join("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/experiments", "inps_exp_COPD_epoch15")
print(in_nii_path, out_nii)
os.makedirs(out_nii, exist_ok=True)
print(f'Identify {len(in_nii_path)} scans (.nii.gz)')

with torch.no_grad():
    generator.eval()
    for nii_path in tqdm(in_nii_path, total = len(in_nii_path)):
        test_dataset = InferenceDataloader(nii_path) #Load the volume into the dataloader 
        test_dataset.load_nii()
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 50, shuffle=False, num_workers=6) #returns the pid, normalized data and the slice index
        converted_scan_idx_slice_map = {}
        for i, data in enumerate(test_dataloader):
            pid = data['pid']
            norm_data = data['normalized_data'].float().to(device) #Data on the device 
            fake_image = generator(norm_data) #fake image generated. this is a tensor which needs to be converted to numpy array
            fake_image_numpy = fake_image.cpu().numpy()
            slice_idx_list = data['slice'].data.cpu().numpy().tolist()
            for idx, slice_index in enumerate(slice_idx_list):
                converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions
        
        nii_file_name = os.path.basename(nii_path)
        converted_image = os.path.join(out_nii, nii_file_name)
        test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
        print(f"{nii_file_name} converted!") 
