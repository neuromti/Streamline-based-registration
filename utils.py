import numpy as np
from dipy.io.streamline import load_tck
from dipy.tracking.utils import density_map
from dipy.io.image import load_nifti


def tck_to_tdi(tck_path, ref_nii_path):
    """Load a .tck file and generate a Tract Density Image (DTI)"""
    reference_data, reference_affine = load_nifti(ref_nii_path)
    reference_shape = reference_data.shape

    tck_file = load_tck(tck_path, ref_nii_path)
    streamlines = tck_file.streamlines

    tdi = density_map(streamlines, reference_affine, reference_shape)

    #print(f"Generatde DTI from {tck_path}. Shape: {tdi.shape}. Max Density: {np.max(tdi)}")

    #save_nifti('tdi-output.nii.gz', tdi.astype('int32'), affine=reference_affine)

    return tdi


def weighted_dice_score(tdi1: np.ndarray, tdi2: np.ndarray, smooth: float = 0.0) -> float:
    """Calculates the Weighted Dice Similarity Coefficient (wDSC) between two TDIs.
    """

    T1_flat = tdi1.ravel()
    T2_flat = tdi2.ravel()

    intersection_sum = np.sum(np.minimum(T1_flat, T2_flat))

    sum_T1 = np.sum(T1_flat)
    sum_T2 = np.sum(T2_flat)

    numerator = 2.0 * intersection_sum + smooth
    denominator = sum_T1 + sum_T2 + smooth

    wdsc = numerator / denominator

    return wdsc