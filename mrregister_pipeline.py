import subprocess
import os
import sys

TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output/mrregister'
TEMPLATE_SUBJECT = '959574'

SUBJECT_IDS = list(filter(lambda x: not x.endswith('.txt') and x != TEMPLATE_SUBJECT, os.listdir(TRACTS_10_DIR)))
TEMPLATE_IMAGE = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
TRACKS_INPUT = os.path.join(TRACTS_10_DIR, TEMPLATE_SUBJECT, "tracts_tck", "CC.tck")

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
        
def process_subject(subject_id):
    """Executes the sequence of transformation commands for a single subject."""

    print(f"\n=============================================")
    print(f"Starting processing for subject: **{subject_id}**")
    print(f"=============================================")

    # Define subject-specific file names
    #subject_difussion_ref = os.path.join(MRI_10_DIR, 'data', f"{subject_id}_Diffusion3TRecommended", f"{subject_id}", "T1w", "Diffusion", "data.nii.gz")
    temp_fod_mif = os.path.join(OUTPUT_DIR, TEMPLATE_SUBJECT, "wmfod.mif")
    moving_fod_mif = os.path.join(OUTPUT_DIR, subject_id, "wmfod.mif")
    
    moving_to_temp_mif = os.path.join(OUTPUT_DIR, subject_id, f"{subject_id}_moving_to_temp.mif")
    temp_to_moving_mif = os.path.join(OUTPUT_DIR, subject_id, f"{subject_id}_temp_to_moving.mif")
    tck_output = os.path.join(OUTPUT_DIR, subject_id, f"{subject_id}_CC_warped.tck")
    
    cmd_step1 = [
        "mrregister", 
        moving_fod_mif, 
        temp_fod_mif,
        "-type", "rigid_affine_nonlinear",
        "-nl_warp", moving_to_temp_mif, temp_to_moving_mif
    ]
    
    run_command(cmd_step1, subject_id)
    print(f"FOD registration completed for subject {subject_id}.")

    cmd_step2 = [
        "tcktransform",
        TRACKS_INPUT,
        moving_to_temp_mif,
        tck_output]
    
    run_command(cmd_step2, subject_id)
    print(f"Deformation applied to tractography file for subject {subject_id}.")
    
    
if __name__ == "__main__":
    if not SUBJECT_IDS:
        print("Error: SUBJECT_IDS list is empty. Please add subject IDs to the list.")
        sys.exit(1)

    for sub_id in SUBJECT_IDS:
        process_subject(sub_id)
        print("\n\nAll subjects processed successfully!")