import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import datetime as dt
import argparse

from PIL import Image
from utils.train import ProgressiveGANTrainer

import matplotlib.pyplot as plt

from IPython import display

def get_options():
    parser = argparse.ArgumentParser()

    # General options. Change all other options in pggan_config.py
    parser.add_argument('--save_iter', default=5000, type=int, help='Number of epochs before saving')
    parser.add_argument('--loss_iter_evaluation', default=200, type=int, help='Number of epochs before saving an image')

    # Optimizer options
    parser.add_argument('--glr', default=1e-3, type=float, help='Learning rate for generator')
    parser.add_argument('--dlr', default=1e-3, type=float, help='Learning rate for discriminator')
    parser.add_argument('--beta1', default=0., type=float, help='Adam optimizer beta1.')
    parser.add_argument('--beta2', default=0.99, type=float, help='Adam optimizer beta2.')

    parser.add_argument('--restore', action='store_true', help='Restore from last checkpoint')
    parser.add_argument('--restore_gdrive', action='store_true', help='Restore from last checkpoint in gdrive')

    opt = parser.parse_args()

    # General options Asserts
    assert opt.save_iter > 0
    assert opt.loss_iter_evaluation > 0

    # Optimizer options asserts
    assert opt.glr > 0
    assert opt.dlr > 0
    assert opt.beta1 >= 0
    assert opt.beta2 >= 0

    return opt

if __name__ == '__main__':
    opt = get_options()

    data_directory = os.path.join(os.getcwd(),'data')
    image_path_pattern = os.path.join(data_directory,'google_pavilion','*.jpeg')

    generator_optimizer = keras.optimizers.Adam(opt.glr ,beta_1=opt.beta1, beta_2=opt.beta2)
    discriminator_optimizer = keras.optimizers.Adam(opt.dlr ,beta_1=opt.beta1, beta_2=opt.beta2)

    pggan_trainer = ProgressiveGANTrainer(datapath=image_path_pattern, 
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator_optimizer=generator_optimizer,
                                        save_iter=opt.save_iter,
                                        loss_iter_evaluation=opt.loss_iter_evaluation
    )

    print(pggan_trainer.modelConfig)

    try:
        from google.colab import drive
        colab = True
        print('Training in colab environement')
    except ModuleNotFoundError:
        colab = False

    pggan_trainer.train(restore=opt.restore, colab=colab, load_from_g_drive=opt.restore_gdrive, g_drive_path = '/content/drive/My Drive/CML')