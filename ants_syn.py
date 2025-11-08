import os 
import shutil
import ants

TRACTS_10_DIR = './Dataset_10_subjects./HCP10_Zenodo'
MRI_10_DIR = './HCP_data'
SUBJECTS = list(filter(lambda x: not x.endswith('.txt'), os.listdir(TRACTS_10_DIR)))
TEMPLATE_SUBJECT = '959574'
TEMPLATE_PATH = os.path.join(MRI_10_DIR, 'data', TEMPLATE_SUBJECT, 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')

"""
# Unzip all the .zip files in MRI_10_DIR
for file in os.listdir(MRI_10_DIR):
    if file.endswith('.zip'):
        zip_path = os.path.join(MRI_10_DIR, file)
        dest_dir = os.path.join(MRI_10_DIR, 'data')  # Remove .zip extension
        os.makedirs(dest_dir, exist_ok=True)
        os.system(f'unzip -o {zip_path} -d {dest_dir}')
"""

# Perform registration for each subject
for subject in SUBJECTS:
    
    if subject == TEMPLATE_SUBJECT:
        continue  # Skip the template subject
    
    print(f"Processing subject: {subject}")
    subject_mri_path = os.path.join(MRI_10_DIR, 'data', subject, 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
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
    dest_dir = os.path.join('output', subject)
    os.makedirs(dest_dir, exist_ok=True)
    ants.image_write(warped_moving_image, os.path.join(dest_dir, f'warped_moving_image_{subject}.nii.gz'))
    print(f"Warped image saved for subject {subject}")

    # Save the inverse transforms and affine matrix
    temp_inv_warp_path = mytx['invtransforms'][1]
    temp_affine_path = mytx['fwdtransforms'][0] 
    
    # Define saving paths
    final_inv_warp_path = os.path.join(dest_dir, f'inv_warp_{subject}.nii.gz')
    final_affine_path = os.path.join(dest_dir, f'affine_transform_{subject}.mat')
    
    # Copy the Inverse Warp Field (.nii.gz) and the Affine Matrix (.mat) to the destination directory
    shutil.copyfile(temp_inv_warp_path, final_inv_warp_path)
    shutil.copyfile(temp_affine_path, final_affine_path)
    print(f"Inverse transforms and affine matrix saved for subject {subject}")


# Run the following commands in terminal for applying the transforms to tractography data
""" 
warpinit data/MNI_2009b_1mm_Brain.nii.gz identity_warp[].nii

for i in {0..2}; do     antsApplyTransforms -d 3 -e 0 -i identity_warp${i}.nii -o mrtrix_temp_to_sub_warp${i}.nii -r data/3102_BL_PD_1/T1_brain.nii.gz -t antsInvWarp.nii.gz -t antsAffine.mat  --default-value 2147483647; done

warpcorrect mrtrix_temp_to_sub_warp[].nii final_temp_to_sub_warp.mif -marker 2147483647

tcktransform data/dbs_tractography.tck final_temp_to_sub_warp.mif output.tck
"""