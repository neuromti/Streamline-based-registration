import subprocess
import os
import sys

TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output/mrregister'
TEMPLATE_SUBJECT = '959574'
SUBJECT_IDS = list(filter(lambda x: not x.endswith('.txt'), os.listdir(TRACTS_10_DIR)))
TEMPLATE_IMAGE = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
DWI_RAW_FILE = "data.nii.gz"
BVECS_FILE = "bvecs"
BVALS_FILE = "bvals"
MASK_FILE = "nodif_brain_mask.nii.gz"

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
        
        
def run_fod_pipeline(subject_id: str):
    """Performs the full FOD estimation processing for a single subject."""
    
    print(f"\n==============================================")
    print(f"Starting FOD pipeline for Subject ID: {subject_id}")
    print(f"==============================================")

    # 1. Define all subject-specific directories and ensure output path exists
    subject_input_dir = os.path.join(MRI_10_DIR, 'data',  f"{subject_id}_Diffusion3TRecommended", f"{subject_id}", 'T1w', "Diffusion")
    subject_output_dir = os.path.join(OUTPUT_DIR , subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # 2. Define all file paths (Inputs are assumed to be in the input directory)
    # Inputs
    dwi_raw_path = os.path.join(subject_input_dir, DWI_RAW_FILE)
    bvecs_path = os.path.join(subject_input_dir, BVECS_FILE)
    bvals_path = os.path.join(subject_input_dir, BVALS_FILE)
    mask_path = os.path.join(subject_input_dir, MASK_FILE)

    # Intermediates / Outputs
    dwi_mif = os.path.join(subject_output_dir, "dwi_preproc.mif")
    wm_response = os.path.join(subject_output_dir, "wm.txt")
    gm_response = os.path.join(subject_output_dir, "gm.txt")
    csf_response = os.path.join(subject_output_dir, "csf.txt")
    response_voxels = os.path.join(subject_output_dir, "response_voxels.mif")
    wmfod_mif = os.path.join(subject_output_dir, "wmfod.nii.gz")
    gm_mif = os.path.join(subject_output_dir, "gm.mif")
    csf_mif = os.path.join(subject_output_dir, "csf.mif")
    
    # Check if necessary input files exist
    if not all(os.path.exists(p) for p in [dwi_raw_path, bvecs_path, bvals_path, mask_path]):
        print(f"Skipping {subject_id}: Missing one or more required input files in {subject_input_dir}")
        return

    # Convert to MRtrix Format ---
    # Command: mrconvert data.nii.gz dwi_preproc.mif -fslgrad bvecs bvals
    cmd_step_1 = [
        "mrconvert",
        str(dwi_raw_path),
        str(dwi_mif),
        "-fslgrad", str(bvecs_path), str(bvals_path)
    ]
    run_command(cmd_step_1, subject_id)


    # Estimate the Multi-Tissue Response Functions (dhollander) ---
    # Command: dwi2response dhollander dwi_preproc.mif wm.txt gm.txt csf.txt -mask nodif_brain_mask.nii.gz -voxels response_voxels.mif
    cmd_step_2 = [
        "dwi2response",
        "dhollander",
        str(dwi_mif),
        str(wm_response),
        str(gm_response),
        str(csf_response),
        "-mask", str(mask_path),
        "-voxels", str(response_voxels)
    ]
    run_command(cmd_step_2, subject_id)


    # Estimate the FODs using MSMT-CSD ---
    # Command: dwi2fod msmt_csd dwi_preproc.mif wm.txt wmfod.mif gm.txt gm.mif csf.txt csf.mif -mask nodif_brain_mask.nii.gz -lmax 8,0,0
    cmd_step_3 = [
        "dwi2fod",
        "msmt_csd",
        str(dwi_mif),
        str(wm_response), str(wmfod_mif),
        str(gm_response), str(gm_mif),
        str(csf_response), str(csf_mif),
        "-mask", str(mask_path),
        "-lmax", "8,0,0" # Commonly used for MSMT-CSD (Lmax 8 for WM, 0 for GM/CSF)
    ]
    run_command(cmd_step_3, subject_id)

    print(f"\n*** Pipeline complete for Subject {subject_id}. FODs saved to: {subject_output_dir} ***")
    
if __name__ == "__main__":
    # Ensure the root output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    

    # --- Start Processing Loop ---
    failed_subjects = []

    for sub_id in SUBJECT_IDS:
        try:
            run_fod_pipeline(sub_id)
        except Exception as e:
            print(f"Processing failed for subject {sub_id}. Check the console output for the specific command error.")
            failed_subjects.append(sub_id)

    if failed_subjects:
        print("\n\n#####################################################")
        print("PIPELINE FINISHED WITH FAILURES:")
        print(f"The following subjects failed to process: {failed_subjects}")
        print("#####################################################")
    else:
        print("\n\nAll subjects processed successfully! ")
