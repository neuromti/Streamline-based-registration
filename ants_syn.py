import os 
import shutil
import ants

TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
OUTPUT_DIR = './output'
SUBJECTS = list(filter(lambda x: not x.endswith('.txt'), os.listdir(TRACTS_10_DIR)))
TEMPLATE_SUBJECT = '959574'
TEMPLATE_PATH = os.path.join(MRI_10_DIR, 'data', f"{TEMPLATE_SUBJECT}_StructuralRecommended", f"{TEMPLATE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')

# Perform registration for each subject
for subject in SUBJECTS:
    
    if subject == TEMPLATE_SUBJECT:
        continue  # Skip the template subject
    
    print(f"Processing subject: {subject}")
    subject_mri_path = os.path.join(MRI_10_DIR, 'data', f"{subject}_StructuralRecommended", f"{subject}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
    fixed_image = ants.image_read(TEMPLATE_PATH)
    moving_image = ants.image_read(subject_mri_path)

    mytx = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='SyN' 
    )
    
    # Apply the transform to the moving image 
    warped_moving_image = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=mytx['fwdtransforms'] ,
        whichtoinvert=[False, False]
    )

    # Save the output
    dest_dir = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(dest_dir, exist_ok=True)
    ants.image_write(warped_moving_image, os.path.join(dest_dir, f'warped_moving_image_{subject}.nii.gz'))
    print(f"Warped image saved for subject {subject}")

    # Save the inverse transforms and affine matrix
    temp_inv_warp_path = mytx['invtransforms'][1]
    temp_affine_path = mytx['invtransforms'][0] 
    
    # Define saving paths
    final_inv_warp_path = os.path.join(dest_dir, f'inv_warp_{subject}.nii.gz')
    final_affine_path = os.path.join(dest_dir, f'affine_transform_{subject}.mat')
    
    # Copy the Inverse Warp Field (.nii.gz) and the Affine Matrix (.mat) to the destination directory
    shutil.copyfile(temp_inv_warp_path, final_inv_warp_path)
    shutil.copyfile(temp_affine_path, final_affine_path)
    print(f"Inverse transforms and affine matrix saved for subject {subject}")
    