import os

MRI_10_DIR = './HCP_10_MRI'  # Directory containing the .zip files


# Unzip all the .zip files in MRI_10_DIR


for file in os.listdir(MRI_10_DIR):
    if file.endswith('.zip'):
        zip_path = os.path.join(MRI_10_DIR, file)
        dest_dir = os.path.join(MRI_10_DIR, 'data', file[:-4])  # Remove .zip extension
        os.makedirs(dest_dir, exist_ok=True)
        os.system(f'unzip -o {zip_path} -d {dest_dir}')



"""
file1 = os.listdir(MRI_10_DIR)[1]
file2 = os.listdir(MRI_10_DIR)[3]

os.system(f'unzip -o {os.path.join(MRI_10_DIR, file1)}')
os.system(f'unzip -o {os.path.join(MRI_10_DIR, file2)}')
"""
#print(len(list(filter(lambda x: x.endswith('.zip'), os.listdir(MRI_10_DIR)))))