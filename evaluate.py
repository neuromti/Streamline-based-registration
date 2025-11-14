from utils  import tck_to_tdi, weighted_dice_score, subsample_streamlines, standardize_streamlines, calculate_bmd
import os
import numpy as np
from dipy.io.streamline import load_tck

TEMPLATE_SUBJECT = '959574'
SUBJECTS_ID = 965771
TRACTS_10_DIR = './HCP10_Tracts'
SUBJECT_IDS = list(filter(lambda x: not x.endswith('.txt') and x != TEMPLATE_SUBJECT, os.listdir(TRACTS_10_DIR)))
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output'
METHODS = ['synthmorph', 'syn']
#METHODS = ['synthmorph']
REFERENCE_NIFTI_PATH = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')

# Define the target number of streamlines for subsampling
TARGET_STREAMLINE_COUNT = 10000
# Define the standard number of points for streamline resampling
STANDARD_POINT_COUNT = 28

for method in METHODS:
    wdsc_results = []
    bmd_results = []
    dest_dir = os.path.join('./output', method)
    print(f"\n\n########## Evaluating method: {method} ##########\n")
    for subject_id in SUBJECT_IDS:
        tracts_moving_path = os.path.join(TRACTS_10_DIR, f"{subject_id}", "tracts_tck", "CC.tck")
        tracts_warped_path = os.path.join(dest_dir, subject_id ,  f"{subject_id}_CC_warped.tck")
        
        # Convert TCK files to Track Density Images (TDI)
        tdi_moving = tck_to_tdi(tracts_moving_path, REFERENCE_NIFTI_PATH)
        tdi_warped = tck_to_tdi(tracts_warped_path, REFERENCE_NIFTI_PATH)
        
        # Calculate Weighted Dice Similarity Coefficient (wDSC)
        wdsc_result = weighted_dice_score(tdi_moving, tdi_warped)
        wdsc_results.append(wdsc_result)
        print(f"Weighted Dice Similarity Coefficient (wDSC) {subject_id}: **{wdsc_result:.4f}**")
        
        # Load streamlines for Bundle Minimum Distance (BMD) calculation
        loadtck_moving = load_tck(tracts_moving_path, REFERENCE_NIFTI_PATH)
        tracts_moving = loadtck_moving.streamlines
        loadtck_warped = load_tck(tracts_warped_path, REFERENCE_NIFTI_PATH)
        tracts_warped = loadtck_warped.streamlines
        
        # Subsample streamlines to target count
        tracts_moving_sub = subsample_streamlines(tracts_moving, TARGET_STREAMLINE_COUNT)
        tracts_warped_sub = subsample_streamlines(tracts_warped, TARGET_STREAMLINE_COUNT)
        
        # Standardize streamlines to have the same number of points
        tracts_moving_std = standardize_streamlines(tracts_moving_sub, STANDARD_POINT_COUNT)
        tracts_warped_std = standardize_streamlines(tracts_warped_sub, STANDARD_POINT_COUNT)
        
        # Calculate Bundle Minimum Distance (BMD)
        bmd_value = calculate_bmd(tracts_moving_std, tracts_warped_std)
        bmd_results.append(bmd_value)
        
        print(f"Bundle Minimum Distance (BMD) {subject_id}: **{bmd_value:.4f}**")
        # write individual results to csv file
        with open(os.path.join(dest_dir, 'results.csv'), 'a') as f:
            f.write(f"{subject_id},{wdsc_result:.4f},{bmd_value:.4f}\n")    

    # Calculate the average wDSC and std across all subjects 
    average_wdsc = np.mean(wdsc_results)
    std_wdsc = np.std(wdsc_results)
    
    # Calculate the average BMD and std across all subjects
    average_bmd = np.mean(bmd_results)
    std_bmd = np.std(bmd_results)
    
    print(f"\n=== Average Weighted Dice Similarity Coefficient (wDSC) across all subjects: **{average_wdsc:.4f}** ===")
    print(f"=== Standard Deviation of Weighted Dice Similarity Coefficient (wDSC) across all subjects: **{std_wdsc:.4f}** ===")
    print(f"\n=== Average Bundle Minimum Distance (BMD) across all subjects: **{average_bmd:.4f}** ===")
    print(f"=== Standard Deviation of Bundle Minimum Distance (BMD) across all subjects: **{std_bmd:.4f}** ===")
    
    # write average results to csv file
    with open(os.path.join(dest_dir, 'results.csv'), 'a') as f:
        f.write(f"Average,{average_wdsc:.4f},{average_bmd:.4f}\n")
        f.write(f"Std,{std_wdsc:.4f},{std_bmd:.4f}\n")