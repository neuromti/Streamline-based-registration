# Generate a csv file which contains that list of path files to mri path, fod path and tck path
import os
import csv 

TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output/mrregister'
TEMPLATE_SUBJECT = '959574'

SUBJECT_IDS = list(filter(lambda x: not x.endswith('.txt') and x != TEMPLATE_SUBJECT, os.listdir(TRACTS_10_DIR)))
#TEMPLATE_IMAGE = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
#TRACKS_INPUT = os.path.join(TRACTS_10_DIR, TEMPLATE_SUBJECT, "tracts_tck", "CC.tck")


def generate_csv(subject_ids, struct_mri_paths, diff_mri_paths, fod_paths, tck_paths, output_csv):
    """
    Generate a CSV manifest file containing paths to MRI, FOD, and TCK files.

    Args:
        struct_mri_paths (list): List of paths to structural MRI files.
        diff_mri_paths (list): List of paths to diffusion MRI files.
        fod_paths (list): List of paths to FOD files.
        tck_paths (list): List of paths to TCK files.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['subject_id', 'struct_mri_path', 'diff_mri_path', 'fod_path', 'tck_path']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for subject_id, struct_mri, diff_mri, fod, tck in zip(subject_ids, struct_mri_paths, diff_mri_paths, fod_paths, tck_paths):
            writer.writerow({'subject_id': subject_id, 'struct_mri_path': struct_mri, 'diff_mri_path': diff_mri, 'fod_path': fod, 'tck_path': tck})
            
if __name__ == "__main__":
    # Example usage:
    struct_mri_files = [os.path.join(MRI_10_DIR, 'data', f"{subject_id}_StructuralRecommended", f"{subject_id}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz') for subject_id in SUBJECT_IDS]
    diff_mri_files = [os.path.join(MRI_10_DIR, 'data', f"{subject_id}_Diffusion3TRecommended", f"{subject_id}", "T1w", "Diffusion", "data.nii.gz") for subject_id in SUBJECT_IDS]
    tck_files = [os.path.join(TRACTS_10_DIR, f"{subject_id}", "tracts_tck", "CC.tck") for subject_id in SUBJECT_IDS]
    fod_files = [os.path.join(OUTPUT_DIR, subject_id, "wmfod.nii.gz") for subject_id in SUBJECT_IDS]
    generate_csv(SUBJECT_IDS, struct_mri_files, diff_mri_files, fod_files, tck_files, 'data_manifest.csv')
    print("CSV manifest file 'data_manifest.csv' generated successfully.")