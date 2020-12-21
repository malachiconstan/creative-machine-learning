import os
import tensorflow as tf
import argparse
import numpy as np

from glob import glob
from PIL import Image
from utils.preprocessing import get_image_dataset, get_cgan_image_datasets
from utils.models import Generator, Discriminator, CGGenerator, CGDiscriminator, get_classifier
from utils.train import train, CycleGANTrainer, ClassifierTrainer

def get_options():
    '''
    Collects all arguments from the command line and checks if they are valid before returning the options

    :return:
        Namespace opt: Contains options as attributes of the object and their values
    '''
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--img_height', default=32, type=int, help='Image Height and Width for classifier and DCGAN. CGAN uses 128x128 default')
    parser.add_argument('--latent_dim', default=100, type=int, help='Dimension of latent dimension')
    parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to train on')
    parser.add_argument('--save_step', default=100, type=int, help='Number of epochs before saving')
    parser.add_argument('--saveimg_step', default=10, type=int, help='Number of epochs before saving an image')

    # Optimizer options
    parser.add_argument('--glr', default=1e-4, type=float, help='Learning rate for generator')
    parser.add_argument('--dlr', default=1e-4, type=float, help='Learning rate for discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam optimizer beta1.')
    parser.add_argument('--beta2', default=0.5, type=float, help='Adam optimizer beta2.')

    # CGAN options
    parser.add_argument('--cgan', action='store_true', help='Run Cycle GAN')
    parser.add_argument('--cgan_restore', action='store_true', help='Restore Cycle GAN from checkpoint')
    parser.add_argument('--restore_gdrive', action='store_true', help='Restore from last checkpoint in gdrive')
    parser.add_argument('--clean_data_dir', action='store_true', help='Remove all images in data dir with less than 128 pixel H/W')

    # Classifier Options
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--classifier', action='store_true', help='Train Classifier')
    parser.add_argument('--infer', action='store_true', help='Infer Data From Trained Classifier')    

    opt = parser.parse_args()

    # General options Asserts
    assert opt.batch_size > 0
    assert opt.img_height > 0
    assert opt.latent_dim > 0
    assert opt.epochs > 0
    assert opt.save_step > 0
    assert opt.saveimg_step > 0

    # Optimizer options asserts
    assert opt.glr > 0
    assert opt.dlr > 0
    assert opt.beta1 > 0
    assert opt.beta2 > 0
    assert not all([opt.cgan, opt.classifier, opt.infer])

    return opt

if __name__ == '__main__':
    opt = get_options()

    # Depending on whether it is running in colab, set the boolean variable colab to be true or false
    try:
        from google.colab import drive
        colab = True
        print('Training in colab environement')
    except ModuleNotFoundError:
        colab = False

    # CGAN Training
    if opt.cgan:
        # Set Data Directory for CGAN Training
        data_directory = os.path.join(os.getcwd(),'data','FACADES_UNPAIRED')

        # Rename files from .jpg to .jpeg
        directories = [
            os.path.join(data_directory,'unpaired_train_A'),
            os.path.join(data_directory,'unpaired_train_B'),
            os.path.join(data_directory,'unpaired_test_A'),
            os.path.join(data_directory,'unpaired_test_B')
        ]

        for directory in directories:
            file_paths = []
            for (dirpath, dirnames, filenames) in os.walk(directory):
                file_paths.extend(filenames)
                break
            for file_path in file_paths:
                file_path = os.path.join(directory, file_path)
                if '.jpg' in os.path.splitext(file_path)[1]:
                    base = os.path.splitext(file_path)[0]
                    os.rename(file_path, base + '.jpeg')

        # Remove all images that are of insufficient Height and Width
        # TODO: To complete such steps before training CGAN
        data_patterns = [
            os.path.join(data_directory,'unpaired_train_A','*.jpeg'),
            os.path.join(data_directory,'unpaired_train_B','*.jpeg'),
            os.path.join(data_directory,'unpaired_test_A','*.jpeg'),
            os.path.join(data_directory,'unpaired_test_B','*.jpeg')
        ]

        # Image height only applicable to CGAN
        IMAGE_HEIGHT=128

        if opt.clean_data_dir:
            for data_dir in data_patterns:
                pic_list = glob(data_dir)
                pic_image_length = len(pic_list)
                count = 0
                for fp in pic_list:
                    shape = np.array(Image.open(fp)).shape
                    if shape[0] < IMAGE_HEIGHT or shape[1] < IMAGE_HEIGHT:
                        count+=1
                        os.remove(fp)
                print(f'Removed {count} images. Left {pic_image_length-count}')

        # Obtain necessary train and test datasets
        train_datasetA =  get_cgan_image_datasets(os.path.join(data_directory,'unpaired_train_A','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=True)
        train_datasetB = get_cgan_image_datasets(os.path.join(data_directory,'unpaired_train_B','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=False)

        test_datasetA =  get_cgan_image_datasets(os.path.join(data_directory,'unpaired_test_A','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=True)
        test_datasetB = get_cgan_image_datasets(os.path.join(data_directory,'unpaired_test_B','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=False)

        # Get Generators and Discriminators
        generator_a2b = CGGenerator()
        generator_b2a = CGGenerator()
        discriminator_a = CGDiscriminator()
        discriminator_b = CGDiscriminator()

        # Build generators and discriminators
        generator_a2b.build((1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3))
        generator_b2a.build((1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3))
        discriminator_a.build((1, IMAGE_HEIGHT,IMAGE_HEIGHT,3))
        discriminator_b.build((1, IMAGE_HEIGHT,IMAGE_HEIGHT,3))

        # Get generator and discriminator optimisers
        generator_a2b_optimizer = tf.keras.optimizers.Adam(opt.glr, beta_1=opt.beta1, beta_2=opt.beta2)
        generator_b2a_optimizer = tf.keras.optimizers.Adam(opt.glr, beta_1=opt.beta1, beta_2=opt.beta2)
        discriminator_a_optimizer = tf.keras.optimizers.Adam(opt.dlr, beta_1=opt.beta1, beta_2=opt.beta2)
        discriminator_b_optimizer = tf.keras.optimizers.Adam(opt.dlr, beta_1=opt.beta1, beta_2=opt.beta2)

        # Get CGAN Trainer
        cgan_trainer = CycleGANTrainer(
            train_datasets = (train_datasetA, train_datasetB),
            test_datasets = (test_datasetA, test_datasetB),
            generators = (generator_a2b, generator_b2a),
            discriminators = (discriminator_a, discriminator_b),
            discriminator_optimizers = (discriminator_a_optimizer, discriminator_b_optimizer),
            generator_optimizers = (generator_a2b_optimizer, generator_b2a_optimizer),
            epochs=opt.epochs,
        )

        # Train CGAN
        cgan_trainer.train(
            restore = opt.cgan_restore,
            colab = colab,
            load_from_g_drive = opt.restore_gdrive,
            save_to_gdrive = True,
            g_drive_path = '/content/drive/My Drive/CML'
        )
    # Classifier Training
    elif opt.classifier:

        print('Training Classifier')
        data_directory = os.path.join(os.getcwd(),'classifier_data')

        # Initialise training dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_directory,
            labels='inferred',
            color_mode='rgb',
            seed=1234,
            validation_split=0.2,
            subset="training",
            image_size=(opt.img_height,opt.img_height),
            batch_size=opt.batch_size,
        )

        # Initialise validation dataset
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_directory,
            labels='inferred',
            color_mode='rgb',
            seed=1234,
            validation_split=0.2,
            subset="validation",
            image_size=(opt.img_height,opt.img_height),
            batch_size=opt.batch_size,
        )

        # Get number of classes within the data directory
        folders = 0
        for _, dirnames, _ in os.walk(data_directory):
            folders += len(dirnames)

        # Initialise Classifier Model
        classifier_net = get_classifier((opt.img_height, opt.img_height, 3), num_classes=folders)

        # Define Learning Rate Schedule
        # TODO: Put this in train.py
        def lr_schedule(epoch, **kwargs):
            '''
            Returns a custom learning rate that decreases as epochs progress.

            :params:
            int epoch: Current epoch

            return: float learning_rate: Learning rate for epoch
            '''
            learning_rate = opt.lr
            if epoch > 25:
                learning_rate = opt.lr/10
            if epoch > 50:
                learning_rate = opt.lr/100
            if epoch > 75:
                learning_rate = opt.lr/1000

            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            return learning_rate

        # Initialise classifier trainer
        trainer = ClassifierTrainer(train_ds, val_ds, classifier_net, tf.keras.optimizers.Adam(learning_rate=opt.lr), lr_schedule)

        # Either infer or train classifier
        if opt.infer:
            infer_dir = os.path.join(os.getcwd(),'classifier_infer_data')
            if not os.path.exists(infer_dir):
                raise FileNotFoundError(infer_dir, ' does not exist!')
            trainer.infer(infer_dir, opt.img_height, opt.img_height)
        else:
            trainer.train(opt.epochs, opt.batch_size)

    else:
        # Train DCGAN
        data_directory = os.path.join(os.getcwd(),'data')

        # Get Train dataset
        train_dataset = get_image_dataset(os.path.join(data_directory,'google_pavilion','*.jpeg'), opt.img_height, opt.img_height, opt.batch_size)

        # Use either normal GAN or upscaled GAN, depending on image resolution
        if opt.img_height == 128:
            print('Using upscaled DCGAN')
            generator = Generator(latent_dim = opt.latent_dim, upscale=True)
            discriminator = Discriminator(upscale=True)
        else:
            generator = Generator(latent_dim = opt.latent_dim)
            discriminator = Discriminator()

        # Build generators and discriminators
        generator.build((opt.batch_size,opt.latent_dim))
        discriminator.build((opt.batch_size,opt.img_height,opt.img_height,3))

        # Optimise generator and discriminator
        generator_optimizer = tf.keras.optimizers.Adam(opt.glr)
        discriminator_optimizer = tf.keras.optimizers.Adam(opt.dlr)

        # Train
        train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, opt.epochs, opt.batch_size, opt.latent_dim, data_directory,False,opt.save_step,opt.saveimg_step)