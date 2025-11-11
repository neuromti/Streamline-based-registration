from utils  import tck_to_tdi, weighted_dice_score
import os
import numpy as np

TEMPLATE_SUBJECT = '959574'
SUBJECTS_ID = 965771
TRACTS_10_DIR = './HCP10_Tracts'
SUBJECT_IDS = list(filter(lambda x: not x.endswith('.txt') and x != TEMPLATE_SUBJECT, os.listdir(TRACTS_10_DIR)))
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output'
METHODS = ['synthmorph']
REFERENCE_NIFTI_PATH = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')


for method in METHODS:
    results = []
    dest_dir = os.path.join('./output', method)
    print(f"\n\n########## Evaluating method: {method} ##########\n")
    for subject_id in SUBJECT_IDS:
        tracts_moving_path = os.path.join(TRACTS_10_DIR, f"{subject_id}", "tracts_tck", "CC.tck")
        tracts_warped_path = os.path.join(dest_dir, subject_id ,  f"{subject_id}_CC_warped.tck")
        tdi_fixed = tck_to_tdi(tracts_moving_path, REFERENCE_NIFTI_PATH)
        tdi_warped = tck_to_tdi(tracts_warped_path, REFERENCE_NIFTI_PATH)
        
        wdsc_result = weighted_dice_score(tdi_fixed, tdi_warped)
        results.append(wdsc_result)
        print(f"Weighted Dice Similarity Coefficient (wDSC) {subject_id}: **{wdsc_result:.4f}**")
        
        # write individual results to csv file
        with open(os.path.join(dest_dir, 'wdsc_results2.csv'), 'a') as f:
            f.write(f"{subject_id},{wdsc_result:.4f}\n")    

    # Calculate and print the average wDSC and std across all subjects and write to csv file    
    average_wdsc = np.mean(results)
    std_wdsc = np.std(results)
    print(f"\n=== Average Weighted Dice Similarity Coefficient (wDSC) across all subjects: **{average_wdsc:.4f}** ===")
    print(f"=== Standard Deviation of Weighted Dice Similarity Coefficient (wDSC) across all subjects: **{std_wdsc:.4f}** ===")
    with open(os.path.join(dest_dir, 'wdsc_results2.csv'), 'a') as f:
        f.write(f"Average,{average_wdsc:.4f}\n")
        f.write(f"Std,{std_wdsc:.4f}\n")