import subprocess
import os
import sys

TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output/synthmorph'
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
    
    moving_img = os.path.join(MRI_10_DIR, 'data', f"{subject_id}_StructuralRecommended", f"{subject_id}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
    fixed_img = TEMPLATE_IMAGE
    dest_dir = os.path.join(OUTPUT_DIR, subject_id)
    os.makedirs(dest_dir, exist_ok=True)

    # Perform registration using SynthMorph
    cmd_step_1 = [
    "./synthmorph",
    "register",
    "-o", os.path.join(dest_dir, f"{subject_id}_moved.nii.gz"),
    "-t",  os.path.join(dest_dir, f"{subject_id}_trans.nii.gz"), 
    moving_img,
    fixed_img,  
    ]
    run_command(cmd_step_1, subject_id)
    print(f"SynthMorph registration completed for subject {subject_id}.")
    
    # Convert displacement field to deformation field
    cmd_step_2 = [
        "warpconvert", 
        os.path.join(dest_dir, f"{subject_id}_trans.nii.gz"), 
        "displacement2deformation", 
        os.path.join(dest_dir, f"{subject_id}_deformation.nii.gz"),
        ]
    run_command(cmd_step_2, subject_id)
    
    # Apply deformation to the tractography file
    cmd_step_3 = [
        "tcktransform",
        TRACKS_INPUT,
        os.path.join(dest_dir, f"{subject_id}_deformation.nii.gz"),
        os.path.join(dest_dir, f"{subject_id}_CC_warped.tck"),
        ]
    run_command(cmd_step_3, subject_id)
    print(f"Deformation applied to tractography file for subject {subject_id}.")
    
if __name__ == "__main__":
    if not SUBJECT_IDS:
        print("Error: SUBJECT_IDS list is empty. Please add subject IDs to the list.")
        sys.exit(1)

    for sub_id in SUBJECT_IDS:
        process_subject(sub_id)

        print("\n\nAll subjects processed successfully!")
    