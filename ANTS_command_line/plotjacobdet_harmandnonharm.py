import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np
import os
from tqdm import tqdm
import seaborn as sns
import pandas as pd

def jacob_det_non_harmonized():
    non_harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/nonharm_ANTS_outputs_exptoinsp_large_images"

    files = sorted(os.listdir(non_harmonized))

    cases_unharm = []
    controls_unharm = []

    cases_files_unharm = []

    for file in tqdm(files):
        if file.endswith("_control"):
            print("Control:", file)
            control = nib.load(os.path.join(non_harmonized, file, "jacobian_det.nii.gz")).get_fdata()
            controls_unharm.append(control.flatten())
        else:
            #Add only 20 cases and keep a track of the name of the files
            print("Case:",file)
            case = nib.load(os.path.join(non_harmonized, file, "jacobian_det.nii.gz")).get_fdata()
            cases_unharm.append(case.flatten())

    #normalize the histograms 
    def normalize_histogram(faltten, num_bins):
        hist, bin_edges = np.histogram(faltten, bins=num_bins)
        normalized_hist = hist/np.sum(hist)
        return normalized_hist, bin_edges

    histograms_cases_unharm = []
    histograms_controls_unharm = []
    num_bins = 50 

    for case, control in zip(cases_unharm, controls_unharm):
        normalized_hist_case_unharm, bin_edges_case_unharm = normalize_histogram(case.flatten(), num_bins)
        normalized_hist_control_unharm, bin_edges_control_unharm = normalize_histogram(control.flatten(), num_bins)
        histograms_cases_unharm.append(normalized_hist_case_unharm)
        histograms_controls_unharm.append(normalized_hist_control_unharm)

    avg_hist_case_unharm = np.mean(histograms_cases_unharm, axis=0)
    avg_hist_control_unharm = np.mean(histograms_controls_unharm, axis=0)

    plt.figure(figsize=(10, 10), facecolor='w')
    # kde for this normalized histogram
    plt.bar(bin_edges_case_unharm[:-1], avg_hist_case_unharm, width=np.diff(bin_edges_case_unharm), color='blue', edgecolor='black', label='COPD_Case', alpha = 0.5)
    plt.bar(bin_edges_control_unharm[:-1], avg_hist_control_unharm, width=np.diff(bin_edges_control_unharm), color='orange', edgecolor='black', label='COPD_Control', alpha = 0.5)
    plt.xlabel('Jacobian Determinant Value', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.xlim(0,3)
    plt.ylabel('Normalized Frequency', fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylim(0,0.2)
    plt.title('Average Normalized Histogram of COPD Cases vs COPD Controls: Non Harmonized', fontsize = 14)
    plt.legend()
    plt.savefig("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/SPIE_paper_figures/quantitative/nonharmonized_jd.tiff", dpi = 300)
    plt.show()

def jacob_det_harmonized():
    harmonized = "/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/data_split/registration_ANTS_command_line/harm_ANTS_outputs_exptoinsp_large_images"

    files = sorted(os.listdir(harmonized))

    cases_harm = []
    controls_harm = []


    for file in tqdm(files):
        if file.endswith("_control"):
            print("Control:", file)
            control = nib.load(os.path.join(harmonized, file, "jacobian_det.nii.gz")).get_fdata()
            controls_harm.append(control.flatten())
        else:
            #Add only 20 cases and keep a track of the name of the files
            print("Case:",file)
            case = nib.load(os.path.join(harmonized, file, "jacobian_det.nii.gz")).get_fdata()
            cases_harm.append(case.flatten())

    #normalize the histograms 
    def normalize_histogram(faltten, num_bins):
        hist, bin_edges = np.histogram(faltten, bins=num_bins)
        normalized_hist = hist/np.sum(hist)
        return normalized_hist, bin_edges

    histograms_cases_harm = []
    histograms_controls_harm = []
    num_bins = 50 

    for case, control in zip(cases_harm, controls_harm):
        normalized_hist_case_harm, bin_edges_case_harm = normalize_histogram(case.flatten(), num_bins)
        normalized_hist_control_harm, bin_edges_control_harm = normalize_histogram(control.flatten(), num_bins)
        histograms_cases_harm.append(normalized_hist_case_harm)
        histograms_controls_harm.append(normalized_hist_control_harm)

    avg_hist_case_harm = np.mean(histograms_cases_harm, axis=0)
    avg_hist_control_harm = np.mean(histograms_controls_harm, axis=0)

    plt.figure(figsize=(10, 10), facecolor='w')
    # kde for this normalized histogram
    plt.bar(bin_edges_case_harm[:-1], avg_hist_case_harm, width=np.diff(bin_edges_case_harm), color='blue', edgecolor='black', label='COPD_Case', alpha = 0.5)
    plt.bar(bin_edges_control_harm[:-1], avg_hist_control_harm, width=np.diff(bin_edges_control_harm), color='orange', edgecolor='black', label='COPD_Control', alpha = 0.5)
    plt.xlabel('Jacobian Determinant Value', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.xlim(0,3)
    plt.ylabel('Normalized Frequency', fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylim(0,0.2)
    plt.title('Average Normalized Histogram of COPD Cases vs COPD Controls: Harmonized', fontsize = 14)
    plt.legend()
    plt.savefig("/nfs/masi/krishar1/SPIE_2025_InhaleExhaleCT/SPIE_paper_figures/quantitative/harmonized_jd.tiff", dpi = 300)
    plt.show()


jacob_det_non_harmonized()
jacob_det_harmonized()