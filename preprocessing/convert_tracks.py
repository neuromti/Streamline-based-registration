import nibabel as nib
import os

os.chdir('preprocessing')

TRK_FOLDER = '../HCP10_Tracts'

SUBJECTS = list(filter(lambda x: not x.endswith('.txt'), os.listdir(TRK_FOLDER)))

# Convert .trk files to .tck files for each subject
for subject in SUBJECTS:
    subject_folder = os.path.join(TRK_FOLDER, subject, 'tracts')
    dest_folder = os.path.join(TRK_FOLDER, subject, 'tracts_tck')
    os.makedirs(dest_folder, exist_ok=True)
    for file in os.listdir(subject_folder):
        if file.endswith('.trk'):
            tract_name = file[:-4]  # Remove .trk extension
            # Load .trk file
            trk = nib.streamlines.load(os.path.join(subject_folder, file))
            # Save as .tck file
            nib.streamlines.save(trk.tractogram,  os.path.join(dest_folder, f'{tract_name}.tck')) 
            print(f'Converted {file} to {tract_name}.tck for subject {subject}')

print('Conversion complete.')