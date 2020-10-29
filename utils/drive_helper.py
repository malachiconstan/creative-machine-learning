import os
import shutil
import zipfile
import glob
from google.colab import drive

def extract_data_g_drive(g_drive_path, mounted = False, local_path='creative-machine-learning', extracting_checkpoints=False, checkpoint_dir = 'pggan_checkpoints'):
    '''
    Helper function to extract the zip file from Google Drive
    :params
        str g_drive_path: preferably os.path, else can also be string showing path to zip file in google drive. zip file should be named data.zip
    '''
    cwd = '/content'
    if not mounted or not os.path.exists('/content/drive/My Drive/'):
        drive.mount('/content/drive')

    data_path = os.path.join(os.path.join(cwd,'drive'),'My Drive',g_drive_path)
    if os.path.exists(data_path):
        shutil.copy(data_path, os.path.join(cwd,'temp.zip'))
        with zipfile.ZipFile(os.path.join(cwd,'temp.zip'),"r") as zip_ref:
            zip_ref.extractall(os.path.join(cwd,local_path))
        print('Extraction complete')
    else:
        raise FileNotFoundError

    if extracting_checkpoints:
        files = glob.glob(f'/content/creative-machine-learning/content/creative-machine-learning/{checkpoint_dir}/*')
        for file_name in files:
            shutil.move(file_name,f'/content/creative-machine-learning/{checkpoint_dir}',os.path.join(f'/content/creative-machine-learning/{checkpoint_dir}',os.path.split(file_name)[1]))

def copy_to_gdrive(local_path, g_drive_path='/content/drive/My Drive/CML/checkpoints.zip'):
    '''
    Helper function to extract folder from colab and save to gdrive. Drive must be mounted
    :params
        str local_path: string to local folder
        str g_drive_path: output zip folder
    '''
    zipf = zipfile.ZipFile(g_drive_path,'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(local_path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()