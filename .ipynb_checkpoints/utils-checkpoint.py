import numpy as np
from dipy.io.streamline import load_tck
from dipy.tracking.utils import density_map
from dipy.io.image import load_nifti
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mdf


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


def subsample_streamlines(streamlines, target_n):
    """Randomly subsamples streamlines to a target count."""
    N = len(streamlines)
    if N <= target_n:
        print(f"Bundle size {N} is already small. Using all.")
        return streamlines
    
    # Generate random indices
    indices = np.random.choice(N, size=target_n, replace=False)
    
    # Select the streamlines
    subsampled_streamlines = streamlines[indices]
    print(f"Subsampled from {N} to {len(subsampled_streamlines)}")
    return subsampled_streamlines

def standardize_streamlines(streamlines, n_points):
    """Resamples all streamlines to have a fixed number of points."""
    if len(streamlines) == 0:
        return []
    
    # set_number_of_points returns a list of new, resampled streamlines
    resampled_streamlines = set_number_of_points(streamlines, n_points)
    
    return resampled_streamlines


def calculate_bmd(tracksA, tracksB):
    """
    Calculates the Bundle-based Minimum Distance (BMD) between two bundles
    of streamlines (tracksA and tracksB) using the MDF metric.

    The BMD is defined as:
    BMD(SA, Sb) = (1/4) * ( (1/A * sum(min_j D(i,j))) + (1/B * sum(min_i D(i,j))) )^2
    where D is the matrix of pairwise MDF distances.

    Parameters
    ----------
    tracksA : list or Streamlines object
        The first bundle of streamlines (SA).
    tracksB : list or Streamlines object
        The second bundle of streamlines (SB).

    Returns
    -------
    bmd_distance : float
        The Bundle-based Minimum Distance (BMD).
    """

    # Calculate the pairwise Minimum Average Direct-Flip (MDF) distance matrix D.
    # D is a matrix of size A x B, where A=len(tracksA) and B=len(tracksB).
    # D[i, j] = MDF(s_i^a, s_j^b)
    # The dipy function calculates the *symmetric* MDF distance for each pair.
    # Note: For MDF, streamlines must have the same number of points.
    D = bundles_distances_mdf(tracksA, tracksB)

    # Get the number of streamlines in each bundle
    A = D.shape[0]
    B = D.shape[1]

    # Check for empty bundles
    if A == 0 or B == 0:
        return 0.0

    # Calculate the average of minimum row values (streamlines in A)
    # This is: (1/A) * sum_{i=1}^{A} min_j D(i,j)
    # np.min(D, axis=1) finds the minimum distance *from* each streamline in A 
    # *to* any streamline in B (i.e., the minimum for each row).
    min_distances_A_to_B = np.min(D, axis=1)
    average_min_A = np.sum(min_distances_A_to_B) / A

    # Calculate the average of minimum column values (streamlines in B)
    # This is: (1/B) * sum_{j=1}^{B} min_i D(i,j)
    # np.min(D, axis=0) finds the minimum distance *from* each streamline in B 
    # *to* any streamline in A (i.e., the minimum for each column).
    min_distances_B_to_A = np.min(D, axis=0)
    average_min_B = np.sum(min_distances_B_to_A) / B

    # Apply the final BMD formula 
    bmd_distance = 0.25 * (average_min_A + average_min_B)**2

    return bmd_distance