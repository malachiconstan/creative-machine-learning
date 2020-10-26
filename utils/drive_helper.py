import os
import shutil
import zipfile
from google.colab import drive

def extract_data_g_drive(g_drive_path, repo_name='creative-machine-learning'):
    '''
    Helper function to extract the zip file from Google Drive
    :params
        str g_drive_path: preferably os.path, else can also be string showing path to zip file in google drive. zip file should be named data.zip
    '''
    cwd = '/content'
    drive.mount('/content/drive')

    data_path = os.path.join(os.path.join(cwd,'drive'),'My Drive',g_drive_path)
    if os.path.exists(data_path):
        shutil.copy(os.path.join(data_path), os.path.join(cwd,'data.zip'))
        with zipfile.ZipFile(os.path.join(cwd,'data.zip'),"r") as zip_ref:
            zip_ref.extractall(os.path.join(cwd,repo_name))
        print('Extraction complete')
    else:
        raise FileNotFoundError

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