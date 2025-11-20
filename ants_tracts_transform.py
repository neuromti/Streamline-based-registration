import subprocess
import os
import sys
import argparse

TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output/syn'
TEMPLATE_SUBJECT = '959574'

SUBJECT_IDS = list(filter(lambda x: not x.endswith('.txt') and x != TEMPLATE_SUBJECT, os.listdir(TRACTS_10_DIR)))
TEMPLATE_IMAGE = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
IDENTITY_WARP_BASE = "identity_warp"
DEFAULT_VALUE = "2147483647"


def run_command(command, subject_id):
    """Executes a shell command and checks for errors."""
    print(f"--- Running command for {subject_id}: {' '.join(command)}")
    try:
        # We use shell=False and pass a list of arguments for better security and portability
        subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Command finished successfully for {subject_id}.")
    except subprocess.CalledProcessError as e:
        print(f"*** ERROR: Command failed for {subject_id}! ***")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1) # Exit the script on the first error

def process_subject(subject_id, bundle_name: str):
    """Executes the sequence of transformation commands for a single subject."""

    print(f"\n=============================================")
    print(f"Starting processing for subject: **{subject_id}**")
    print(f"=============================================")
    # Define input track file
    TRACKS_INPUT = os.path.join(TRACTS_10_DIR, TEMPLATE_SUBJECT, "tracts_tck", f"{bundle_name}.tck")

    # Define subject-specific file names
    subject_t1_ref = os.path.join(MRI_10_DIR, 'data', f"{subject_id}_StructuralRecommended", f"{subject_id}", "T1w", "T1w_acpc_dc_restore_brain.nii.gz")
    mrtrix_temp_to_sub_warp_base = "mrtrix_temp_to_sub_warp"
    final_warp_mif = os.path.join(OUTPUT_DIR, subject_id, f"{subject_id}_final_temp_to_sub_warp.mif")
    tck_output = os.path.join(OUTPUT_DIR, subject_id, f"{subject_id}_{bundle_name}_warped.tck")

    # Creates identity_warp0.nii, identity_warp1.nii, identity_warp2.nii
    if not os.path.exists(final_warp_mif):
        run_command(
            ["warpinit", TEMPLATE_IMAGE,  os.path.join(OUTPUT_DIR, subject_id, f"{IDENTITY_WARP_BASE}[].nii"), "-force"],
            subject_id
        )
    
    
        # Apply transform loop
        ANTs_INVERSE_WARP = os.path.join(OUTPUT_DIR, subject_id, f"inv_warp_{subject_id}.nii.gz")
        ANTs_AFFINE_MAT = os.path.join(OUTPUT_DIR, subject_id, f"affine_transform_{subject_id}.mat")
        for i in range(3):
            input_warp = os.path.join(OUTPUT_DIR, subject_id, f"{IDENTITY_WARP_BASE}{i}.nii")
            output_warp = os.path.join(OUTPUT_DIR, subject_id, f"{mrtrix_temp_to_sub_warp_base}{i}.nii")
            
            run_command(
                [
                    "antsApplyTransforms", 
                    "-d", "3", 
                    "-e", "0", 
                    "-i", input_warp, 
                    "-o", output_warp, 
                    "-r", subject_t1_ref,
                    "-t", ANTs_INVERSE_WARP, 
                    "-t", ANTs_AFFINE_MAT, 
                    "--default-value", DEFAULT_VALUE
                ],
                f"{subject_id} (antsApplyTransforms {i})"
            )
    
        
        # warpcorrect mrtrix_temp_to_sub_warp[].nii final_temp_to_sub_warp.mif -marker 2147483647
        run_command(
        [
            "warpcorrect", 
            #f"{mrtrix_temp_to_sub_warp_base}[].nii", 
            os.path.join(OUTPUT_DIR, subject_id, f"{mrtrix_temp_to_sub_warp_base}[].nii"),
            final_warp_mif, 
            "-marker", DEFAULT_VALUE
        ],
        subject_id
    )
        
    # Perform tck transform
    run_command(
        [
            "tcktransform", 
            TRACKS_INPUT, 
            final_warp_mif, 
            tck_output
        ],
    subject_id
)
    
    # Remove intermediate files
    print(f"Cleaning up intermediate files for {subject_id}...")
    for i in range(3):
        input_warp = os.path.join(OUTPUT_DIR, subject_id, f"{IDENTITY_WARP_BASE}{i}.nii")
        output_warp = os.path.join(OUTPUT_DIR, subject_id, f"{mrtrix_temp_to_sub_warp_base}{i}.nii")
        if os.path.exists(input_warp):
            os.remove(input_warp)
        if os.path.exists(output_warp):
            os.remove(output_warp)
    print(f"Cleanup complete for {subject_id}.")
    
    
    print(f"Finished processing for subject: **{subject_id}**")
    print(f"Output track file: {tck_output}")
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Apply ANTs transformations to tractography data for multiple subjects.")
    ap.add_argument("--bundle", type=str, default="CC", help="Name of the bundle to process (default: CC)")
    args = ap.parse_args()
    bundle_name = args.bundle
    print( f"Processing bundle: {bundle_name}" )
    if not SUBJECT_IDS:
        print("Error: SUBJECT_IDS list is empty. Please add subject IDs to the list.")
        sys.exit(1)

    for sub_id in SUBJECT_IDS:
        process_subject(sub_id, bundle_name)

        print("\n\nAll subjects processed successfully!")