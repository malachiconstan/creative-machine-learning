import os
import shutil
import zipfile
from google.colab import drive

def extract_data_g_drive(g_drive_path):
    '''
    Helper function to extract the zip file from Google Drive
    :params
        str g_drive_path: preferably os.path, else can also be string showing path to zip file in google drive. zip file should be named data.zip
    '''
    cwd = os.getcwd()
    drive.mount(cwd)

    data_path = os.path.join(cwd,'My Drive',g_drive_path)
    if os.path.exists(data_path):
        shutil.copy(os.path.join(data_path), os.path.join(cwd,'data.zip'))
        with zipfile.ZipFile(os.path.join(cwd,'data.zip'),"r") as zip_ref:
            zip_ref.extractall(cwd)
        print('Extraction complete')
    else:
        raise FileNotFoundError