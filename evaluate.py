from utils  import tck_to_tdi, weighted_dice_score
import os
#import numpy as np

TEMPLATE_SUBJECT = '959574'
SUBJECTS_ID = 965771
TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output'

TRACTOGRAM_MOVING_PATH = os.path.join(TRACTS_10_DIR, f"{SUBJECTS_ID}", "tracts_tck", "CC.tck")
TRACTOGRAM_WARPED_PATH = os.path.join(OUTPUT_DIR,  f"{SUBJECTS_ID}", f"{SUBJECTS_ID}_CC_warped.tck")
REFERENCE_NIFTI_PATH = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')

tdi_fixed = tck_to_tdi(TRACTOGRAM_MOVING_PATH, REFERENCE_NIFTI_PATH)
tdi_warped = tck_to_tdi(TRACTOGRAM_WARPED_PATH, REFERENCE_NIFTI_PATH)

wdsc_result = weighted_dice_score(tdi_fixed, tdi_warped)

print("\n--- Similarity Evaluation ---")
#print(f"TDI Moving Streamline Count Sum (proxy for total density): {np.sum(tdi_fixed):.2f}")
#print(f"TDI Warped Streamline Count Sum (proxy for total density): {np.sum(tdi_warped):.2f}")
print(f"Weighted Dice Similarity Coefficient (wDSC): **{wdsc_result:.4f}**")