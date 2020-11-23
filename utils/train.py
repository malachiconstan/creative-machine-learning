import time
import os
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from PIL import Image
from IPython import display
from copy import deepcopy
from glob import glob

from utils.configs.pggan_config import _C
from utils.config import BaseConfig, getConfigFromDict, getDictFromConfig
from utils.models import ProgressiveGAN
from utils.preprocessing import get_image_dataset, process_path
from utils.losses import WGANGPGradientPenalty, cggan_discriminator_loss, cggan_generator_loss, identity_loss, cycle_loss

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([real_output.shape[0],1],0.7,1.2), real_output, from_logits=True) # set noise to 1
    fake_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0,0.3), fake_output, from_logits=True) # set noise to 0
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0.7,1.2), fake_output, from_logits=True) # set noise to 1

@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, latent_dim, batch_size, sdis_loss, sgen_loss, sdis_acc):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        sdis_loss(disc_loss)
        sgen_loss(gen_loss)
        sdis_acc(tf.ones_like(real_output), real_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# def generate_and_save_images(model, epoch, test_input, output_dir):
#     # Notice `training` is set to False.
#     # This is so all layers run in inference mode (batchnorm).
#     predictions = model(test_input, training=False)
#     fig = plt.figure(figsize=(10,10))

#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i+1)
#         plt.imshow(predictions[i, :, :, :]* 0.5 + 0.5) # map from range(-1,1) to range(0,1)

#         plt.axis('off')
#     plt.savefig(os.path.join(output_dir,f'image_at_epoch_{epoch:04d}.png'))
#     plt.close()

def generate_and_save_images(model, epoch, test_input, file_writer):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = predictions[:, :, :, :]* 0.5 + 0.5

    with file_writer.as_default():
        tf.summary.image('Generated Images', predictions, max_outputs=16, step=epoch)

def display_image(epoch_no, output_dir):
    return Image.open(output_dir,f'image_at_epoch_{epoch:04d}.png')

def train(dataset,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        epochs,
        batch_size,
        latent_dim,
        data_directory,
        restore=False,
        save_step=100,
        saveimg_step=10):

    seed = tf.random.normal([16,latent_dim])
    log_dir = os.path.join(os.getcwd(), 'logs')
    output_dir = os.path.join(os.getcwd(), 'outputs')
    checkpoint_path = os.path.join(os.getcwd(),'checkpoints')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    sgen_loss = tf.keras.metrics.Mean('sgen_loss', dtype=tf.float32)
    sdis_loss = tf.keras.metrics.Mean('sdis_loss', dtype=tf.float32)
    sdis_acc = tf.keras.metrics.BinaryAccuracy('sdis_acc')

    current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")
    gen_log_dir = os.path.join(log_dir,'gradient_tape',current_time,'gen')
    dis_log_dir = os.path.join(log_dir,'gradient_tape',current_time,'dis')

    gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
    dis_summary_writer = tf.summary.create_file_writer(dis_log_dir)

    if restore:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("Restored from epoch{}".format(int(checkpoint.step)))
        add_step=int(checkpoint.step)
        print("Restore")
    else:
        add_step=0
        print("Fresh")

    for epoch in range(epochs):

        if restore:
            step=int(checkpoint.step)+epoch
        else:
            step=epoch

        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, latent_dim, batch_size, sdis_loss, sgen_loss, sdis_acc)

        with gen_summary_writer.as_default():
            tf.summary.scalar('sgen_loss', sgen_loss.result(), step=step)

        with dis_summary_writer.as_default():
            tf.summary.scalar('sdis_loss', sdis_loss.result(), step=step)
            tf.summary.scalar('sdis_acc', sdis_acc.result(), step=step)

        display.clear_output(wait=True)
        if (epoch + 1 + add_step)%saveimg_step==0:
            generate_and_save_images(generator,epoch,seed,gen_summary_writer)

        if (epoch + 1) % save_step == 0:
            checkpoint.step.assign_add(save_step)
            checkpoint.save(file_prefix = checkpoint_path)
            print(f'Checkpoint Step: {int(checkpoint.step)}')
        template = 'Epoch {}, Generator Loss: {}, Discriminator Loss: {}, Discriminator Accuracy: {}'
        print (template.format(epoch+1,
                                sgen_loss.result(),
                                sdis_loss.result(),
                                sdis_acc.result()))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        sgen_loss.reset_states()
        sgen_loss.reset_states()
        sdis_loss.reset_states()

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epoch,seed,gen_summary_writer)

class ClassifierTrainer(object):
    def __init__(self,
                train_dataset,
                validation_dataset,
                model,
                optimizer,
                lr_schedule
                ):

        # Define Directories
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = os.path.join(os.getcwd(),'classifier_logs',current_time)
        self.checkpoint_dir = os.path.join(os.getcwd(),'classifier_checkpoints')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)

        # Define Checkpoint
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.checkpoint_dir,'cp.ckpt'), verbose=1, save_weights_only=True, save_freq = 'epoch')

        self.model = model
        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # Define Learning Rate Scheduler
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()

        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        self.model.compile(
            optimizer = self.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ["accuracy"]
        )
    
    def train(self,
            epochs=10,
            batch_size=32
            ):

        self.history = self.model.fit(self.train_dataset,
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[self.cp_callback, self.tensorboard_callback],
                                validation_data=self.validation_dataset)

        print('Training Completed')

    def infer(self,
            infer_datadir,
            img_height,
            img_width,
            classnames
            ):

        self.model.load_weights(os.path.join(self.checkpoint_dir,'cp.ckpt'))
        file_paths = glob(os.path.join(infer_datadir,'*.jpeg'))
        test_pred = tf.stack([process_path(file,img_height,img_width,False,False) for file in file_paths])

        preds = tf.nn.softmax(self.model(test_pred),axis=1).numpy()
        df_preds = pd.DataFrame(preds)
        df_preds.index = [os.path.split(fp)[1] for fp in file_paths]
        df_preds.columns = classnames

        df_preds.to_csv('predictions.csv')

        print('Inference Completed')

class CycleGANTrainer(object):
    def __init__(self,
                train_datasets,
                test_datasets,
                generators,
                discriminators,
                discriminator_optimizers,
                generator_optimizers,
                epochs=40,
                save_epoch=1,
                model_label="Cgan"
                ):

        # Define Directories
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

        self.log_dir = os.path.join(os.getcwd(),'cgan_logs')
        self.gen_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'gen')
        self.dis_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'dis')
        self.checkpoint_dir = os.path.join(os.getcwd(),'cgan_checkpoints')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Datasets
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

        # Optimizers and Models
        self.generator_optimizers = generator_optimizers
        self.discriminator_optimizers = discriminator_optimizers
        self.generators = generators
        self.discriminators = discriminators

        # Losses
        self.generator_loss = cggan_generator_loss
        self.discriminator_loss = cggan_discriminator_loss
        self.cycle_loss = cycle_loss
        self.identity_loss = identity_loss

        # Hyperparams
        self.epochs = epochs
        self.save_epoch = save_epoch

        # Logging
        self.metrics = dict(
            discriminator_a_loss = tf.keras.metrics.Mean('discriminator_a_loss', dtype=tf.float32),
            discriminator_b_loss = tf.keras.metrics.Mean('discriminator_b_loss', dtype=tf.float32),
            generator_a2b_loss = tf.keras.metrics.Mean('generator_a2b_loss', dtype=tf.float32),
            generator_b2a_loss = tf.keras.metrics.Mean('generator_b2a_loss', dtype=tf.float32),
        )

        self.gen_summary_writer = tf.summary.create_file_writer(self.gen_log_dir)
        self.dis_summary_writer = tf.summary.create_file_writer(self.dis_log_dir)

        # Checkpoints
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                        generator_a2b=self.generators[0],
                                        generator_b2a=self.generators[1],
                                        discriminator_a=self.discriminators[0],
                                        discriminator_b=self.discriminators[1],
                                        generator_a2b_optimizer=self.generator_optimizers[0],
                                        generator_b2a_optimizer=self.generator_optimizers[1],
                                        discriminator_a_optimizer=self.discriminator_optimizers[0],
                                        discriminator_b_optimizer=self.discriminator_optimizers[1]
                                        )

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,self.checkpoint_dir,max_to_keep=5)

    def generate_images(self, model, test_input, epoch, output_dir):
        prediction = model(test_input)
            
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig(output_dir+'/training_{}.png'.format(epoch)) 
        plt.show()

    def train(self, restore=False, colab=False, load_from_g_drive=False, save_to_gdrive=True, g_drive_path = '/content/drive/My Drive/CML'):

        if restore:
            if colab and load_from_g_drive:
                from utils.drive_helper import extract_data_g_drive
                extract_data_g_drive('CML/cgan_checkpoints.zip', mounted=True, extracting_checkpoints=True)
            
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.checkpoint_manager.latest_checkpoint:
                print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
            else:
                raise Exception('Cannot find checkpoint')
        
        for epoch in range(self.epochs):
            self.checkpoint.step.assign_add(1)
            start = time.time()
            for img_a, img_b in tf.data.Dataset.zip(self.train_datasets):
                self.train_step(img_a,img_b)
            
            # Log Losses
            with self.gen_summary_writer.as_default():
                tf.summary.scalar('generator_a2b_loss', self.metrics['generator_a2b_loss'].result(), step=epoch)
                tf.summary.scalar('generator_b2a_loss', self.metrics['generator_b2a_loss'].result(), step=epoch)

            with self.dis_summary_writer.as_default():
                tf.summary.scalar('discriminator_a_loss', self.metrics['discriminator_a_loss'].result(), step=epoch)
                tf.summary.scalar('discriminator_b_loss', self.metrics['discriminator_b_loss'].result(), step=epoch)

            # Save Images
            img_b_from_a = self.generators[0](img_a, training=False)
            img_b_from_a = img_b_from_a[:, :, :, :]* 0.5 + 0.5
            img_a = img_a[:, :, :, :]* 0.5 + 0.5

            img_a_from_b = self.generators[1](img_b, training=False)
            img_a_from_b = img_a_from_b[:, :, :, :]* 0.5 + 0.5
            img_b = img_b[:, :, :, :]* 0.5 + 0.5


            with self.gen_summary_writer.as_default():
                tf.summary.image('Image A', img_a, max_outputs=5, step=epoch)
                tf.summary.image('Image A -> B', img_b_from_a, max_outputs=5, step=epoch)
                tf.summary.image('Image B', img_b, max_outputs=5, step=epoch)
                tf.summary.image('Image B -> A', img_a_from_b, max_outputs=5, step=epoch)

            if epoch % self.save_epoch == 0:
                save_path = self.checkpoint_manager.save()
                print('Checkpoint step at: ',int(self.checkpoint.step))
                print(f"Saved checkpoint for step {int(self.checkpoint.step)}: {save_path}")

                if colab and save_to_gdrive:
                    from utils.drive_helper import copy_to_gdrive

                    if not os.path.exists(g_drive_path):
                        if not os.path.exists('/content/drive/My Drive/'):
                            raise NotADirectoryError('Drive not mounted')
                        os.makedirs(g_drive_path)

                    checkpoint_path = os.path.join(g_drive_path,'cgan_checkpoints.zip')
                    logs_path = os.path.join(g_drive_path,'cgan_logs.zip')

                    copy_to_gdrive(local_path=self.checkpoint_dir, g_drive_path=checkpoint_path)
                    copy_to_gdrive(local_path=self.log_dir, g_drive_path=logs_path)
                    print('Checkpoints Saved to ',checkpoint_path)
                    print('Logs Saved to ',logs_path)

            # Reset Losses
            for k in self.metrics:
                self.metrics[k].reset_states()

            print(f'Epoch {epoch} took {time.time()-start:.3f} sec')

    @tf.function
    def train_step(self, real_a,real_b):
        with tf.GradientTape(persistent=True) as tape:

            fake_b = self.generators[0](real_a,training=True)
            cycled_a = self.generators[1](fake_b,training=True)

            fake_a = self.generators[1](real_b,training=True)
            cycled_b = self.generators[0](fake_a,training=True)

            same_a = self.generators[1](real_a,training=True)
            same_b = self.generators[0](real_b,training=True)

            disc_real_a = self.discriminators[0](real_a,training=True)
            disc_real_b = self.discriminators[1](real_b,training=True)

            disc_fake_a = self.discriminators[0](fake_a,training=True)
            disc_fake_b = self.discriminators[1](fake_b,training=True)

            gen_a2b_loss = self.generator_loss(disc_fake_b)
            gen_b2a_loss = self.generator_loss(disc_fake_a)

            total_cycle_loss = self.cycle_loss(real_a,cycled_a) + self.cycle_loss(real_b,cycled_b)

            # Total losses for all 4 
            total_gen_a2b_loss = gen_a2b_loss + total_cycle_loss + self.identity_loss(real_b,same_b)
            total_gen_b2a_loss = gen_b2a_loss + total_cycle_loss + self.identity_loss(real_a,same_a)

            disc_a_loss = self.discriminator_loss(disc_real_a,disc_fake_a)
            disc_b_loss = self.discriminator_loss(disc_real_b,disc_fake_b)

            # Log losses
            self.metrics['discriminator_a_loss'](disc_a_loss)
            self.metrics['discriminator_b_loss'](disc_b_loss)
            self.metrics['generator_a2b_loss'](total_gen_a2b_loss)
            self.metrics['generator_b2a_loss'](total_gen_b2a_loss)
        
        # Calculate graidents for all 4
        gen_a2b_grad = tape.gradient(total_gen_a2b_loss,self.generators[0].trainable_variables)
        gen_b2a_grad = tape.gradient(total_gen_b2a_loss,self.generators[1].trainable_variables)
        disc_a_grad = tape.gradient(disc_a_loss,self.discriminators[0].trainable_variables)
        disc_b_grad = tape.gradient(disc_b_loss,self.discriminators[1].trainable_variables)

        # Apply graidents to all 4 optimizer
        self.generator_optimizers[0].apply_gradients(zip(gen_a2b_grad,self.generators[0].trainable_variables))
        self.generator_optimizers[1].apply_gradients(zip(gen_b2a_grad,self.generators[1].trainable_variables))
        self.discriminator_optimizers[0].apply_gradients(zip(disc_a_grad,self.discriminators[0].trainable_variables))
        self.discriminator_optimizers[1].apply_gradients(zip(disc_b_grad,self.discriminators[1].trainable_variables))

class ProgressiveGANTrainer(object):
    """
    A class managing a progressive GAN training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """

    def __init__(self,
                config,
                discriminator_optimizer,
                generator_optimizer,
                loss_iter_evaluation=200,
                save_iter=5000,
                model_label="PGGAN",
                ):
        """
        Args:
            - pathdb (string): path to the directorty containing the image
                               dataset
            - useGPU (bool): set to True if you want to use the available GPUs
                             for the training procedure
            - visualisation (module): if not None, a visualisation module to
                                      follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the
                                        model'sloss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
                              (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the checkpoints
                                      should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary. See std_p_gan_config.py
                                   for all the possible options
            - numWorkers (int): number of GOU to use. Will be set to one if not
                                useGPU
            - stopOnShitStorm (bool): should we stop the training if a diverging
                                     behavior is detected ?
        """

        # Define directories
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

        self.datapath = config.datapath
        self.log_dir = os.path.join(os.getcwd(),'pggan_logs')
        self.img_dir = os.path.join(os.getcwd(),'pggan_imgs')
        self.gen_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'gen')
        self.dis_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'dis')
        self.checkpoint_dir = os.path.join(os.getcwd(),'pggan_checkpoints')
        self.model_save_dir = os.path.join(os.getcwd(),'pggan_weights')
        self.train_config_path = os.path.join(self.checkpoint_dir, f'{model_label}_' + "_train_config.json")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.config = config

        # Intern state
        self.start_resolution = config.resolution
        self.stop_resolution = config.stop_resolution
        self.start_epoch = config.start_epoch
        self.epochs = config.epochs
        self.batch_size = self.calculate_batch_size(config.resolution)
        self.latent_dim = config.latent_dim
        self.overall_steps = 0

        self.initialise_model()

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        # Checkpoints
        self.model_label = model_label
        self.save_iter = save_iter
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
            epoch = tf.Variable(self.start_epoch),
            resolution = tf.Variable(self.start_resolution),
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.model.Generator,
            discriminator=self.model.Discriminator
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # Logging
        self.loss_iter_evaluation = loss_iter_evaluation
        self.metrics = dict(
            discriminator_wasserstein_loss_real = tf.keras.metrics.Mean('discriminator_wasserstein_loss_real', dtype=tf.float32),
            discriminator_wasserstein_loss_fake = tf.keras.metrics.Mean('discriminator_wasserstein_loss_fake', dtype=tf.float32),
            discriminator_wasserstein_gradient_penalty = tf.keras.metrics.Mean('discriminator_wasserstein_gradient_penalty', dtype=tf.float32),
            generator_wasserstein_loss = tf.keras.metrics.Mean('generator_wasserstein_loss', dtype=tf.float32),
            # discriminator_epsilon_loss = tf.keras.metrics.Mean('discriminator_epsilon_loss', dtype=tf.float32),
            discriminator_loss = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32),
            generator_loss = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32),
            alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)
        )

        self.gen_summary_writer = tf.summary.create_file_writer(self.gen_log_dir)
        self.dis_summary_writer = tf.summary.create_file_writer(self.dis_log_dir)

        # Test out the Generator
        sample_noise = tf.random.normal((9, config.latent_dim), seed=0)
        self.generate_and_save_images(0, sample_noise, figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)

        # Create many tf functions
        # res = self.start_resolution
        # self.discriminator_train_steps = dict()
        # self.generator_train_steps = dict()
        # while res <= self.stop_resolution:
        #     self.discriminator_train_steps[str(res)] = deepcopy(self.discriminator_train_step)
        #     self.generator_train_steps[str(res)] = deepcopy(self.generator_train_step)
        #     res *= 2
        # print('Created training steps')

    def initialise_model(self, resolution=None):
        if resolution is None:
            resolution = self.start_resolution
        print(f'Initialising model with resolution {resolution}')
        self.model = ProgressiveGAN(
            resolution,
            self.config.latent_dim,
            self.config.leaky_relu_leak,
            self.config.kernel_initializer,
            self.config.output_activation
        )

    def generate_and_save_images(self, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
        # Test input is a list include noise and label
        predictions = self.model(test_input)
        fig = plt.figure(figsize=figure_size)
        for i in range(predictions.shape[0]):
            axs = plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(predictions[i] * 0.5 + 0.5)
            plt.axis('off')
        if save:
            plt.savefig(os.path.join(self.img_dir, '{}x{}_image_at_epoch_{:04d}.png'.format(predictions.shape[1], predictions.shape[2], epoch)))
        plt.close()

    def save_check_point(self, resolution, verbose=True, save_to_gdrive=True, g_drive_path = '/content/drive/My Drive/CML'):
        """
        Save a checkpoint at the given directory. Please not that the basic
        configuration won't be saved.
        This function produces 2 files:
        outDir/outLabel_tmp_config.json -> temporary config
        outDir/outLabel -> networks' weights
        """
        save_path = self.checkpoint_manager.save()
        if verbose:
            print('Checkpoint step at: ',int(self.checkpoint.step))
            print(f"Saved checkpoint for step {int(self.checkpoint.step)}: {save_path}")
        
        # After transition mandatory save and load

        if save_to_gdrive:
            from utils.drive_helper import copy_to_gdrive

            if not os.path.exists(g_drive_path):
                if not os.path.exists('/content/drive/My Drive/'):
                    raise NotADirectoryError('Drive not mounted')
                os.makedirs(g_drive_path)

            checkpoint_path = os.path.join(g_drive_path,'checkpoints.zip')
            logs_path = os.path.join(g_drive_path,'logs.zip')

            copy_to_gdrive(local_path=self.checkpoint_dir, g_drive_path=checkpoint_path)
            copy_to_gdrive(local_path=self.log_dir, g_drive_path=logs_path)
            print('Checkpoints Saved to ',checkpoint_path)
            print('Logs Saved to ',logs_path)

    def load_saved_training(self, load_from_g_drive=False):
        """
        Load a given checkpoint.
        """ 
        if self.colab and load_from_g_drive:
            from utils.drive_helper import extract_data_g_drive
            extract_data_g_drive('CML/checkpoints.zip', mounted=True, extracting_checkpoints=True)
            print('Extracted checkpoints from colab')

        # Get resolution from latest checkpoint
        self.start_resolution = tf.train.load_variable(self.checkpoint_dir, 'resolution/.ATTRIBUTES/VARIABLE_VALUE')
        self.overall_steps = tf.train.load_variable(self.checkpoint_dir, 'step/.ATTRIBUTES/VARIABLE_VALUE')
        self.start_epoch = tf.train.load_variable(self.checkpoint_dir, 'epoch/.ATTRIBUTES/VARIABLE_VALUE')
        self.initialise_model()
        
        # Load saved checkpoint
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
        
        print(f'Start training from {self.start_resolution}x{self.start_resolution} at epoch: {self.start_epoch}, step: {self.overall_steps}')

    @staticmethod
    def calculate_batch_size(resolution):
        if resolution in [4,8,16,32,64]:
            return 16
        elif resolution == 128:
            return 8
        elif resolution == 256:
            return 4
        elif resolution == 512:
            return 3
        else:
            raise NotImplementedError(f'{resolution} is not implemented')
    
    @property
    def alpha(self):
        """
        Get alpha value
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """
        if value < 0 or value > 1:
            raise ValueError("alpha must be in [0,1]")
        
        if hasattr(self, 'alpha'):
            if self.alpha != value:
                self.model.alpha = value
                self._alpha = value
        else:
            self.model.alpha = value
            self._alpha = value

    def train(self, restore=False, colab=False, load_from_g_drive=False, verbose=False, g_drive_path = '/content/drive/My Drive/CML'):
        """
        Launch the training. This one will stop if a divergent behavior is
        detected.
        Returns:
            - True if the training completed
            - False if the training was interrupted due to a divergent behavior
        """
        self.colab = colab
        self.train_start_time = time.time()
        self.g_drive_path = g_drive_path

        if restore:
            self.load_saved_training(load_from_g_drive=load_from_g_drive)

        resolution = self.start_resolution
        while resolution <= self.stop_resolution:
            print(f'Size {resolution}x{resolution} training begins')

            # Define specific paths
            # self.temp_config_path = os.path.join(self.checkpoint_dir, f'{self.model_label}_{scale}_' + "_tmp_config.json")
            self.checkpoint.resolution.assign(resolution)
            self.resolution_batch_size = self.calculate_batch_size(resolution)
            
            # Get train dataset at the correct image scale
            train_dataset = get_image_dataset(self.datapath,
                                            img_height=resolution,
                                            img_width=resolution,
                                            batch_size=self.resolution_batch_size,
                                            normalize=True)
            
            if verbose:
                print(f'Dataset for resolution {resolution}x{resolution} obtained')
                print('Dataset Length: ', len(train_dataset))

            training_steps = np.ceil(len(train_dataset) / self.batch_size)
            # Fade in half of switch_res_every_n_epoch epoch, and stablize another half
            self.resolution_alpha_increment = 1. / (self.epochs / 2 * training_steps)
            self.alpha = min(1., (self.start_epoch - 1) % self.epochs * training_steps * self.resolution_alpha_increment)

            assert self.start_epoch <= self.epochs, f'Start epochs {self.start_epoch} should be less than epochs: {self.epochs}'
            for epoch in range(self.start_epoch, self.epochs + 1):
                self.train_epoch(train_dataset, resolution, epoch, verbose=verbose)

            # self.save_check_point(resolution, verbose=True, save_to_gdrive=self.colab, g_drive_path = self.g_drive_path)
            # self.model.Generator.save_weights(os.path.join(self.model_save_dir, f'{resolution}x{resolution}_generator.h5'))
            # self.model.Discriminator.save_weights(os.path.join(self.model_save_dir, f'{resolution}x{resolution}_discriminator.h5'))
            # Add scale
            if resolution != self.stop_resolution:
                self.model.double_resolution()

            resolution *= 2
            # self.initialise_model(resolution)
            # self.load_saved_training(load_from_g_drive=load_from_g_drive)
                    
            # if os.path.isfile(os.path.join(self.model_save_dir, f'{resolution//2}x{resolution//2}_generator.h5')):
            #     self.model.Generator.load_weights(os.path.join(self.model_save_dir, f'{resolution//2}x{resolution//2}_generator.h5'), by_name=True)
            #     print("generator loaded")
            # if os.path.isfile(os.path.join(self.model_save_dir, f'{resolution//2}x{resolution//2}_generator.h5')):
            #     self.model.Discriminator.load_weights(os.path.join(self.model_save_dir, f'{resolution//2}x{resolution//2}_discriminator.h5'), by_name=True)
            #     print("discriminator loaded")

        return True

    def train_epoch(self,
                    dataset,
                    resolution,
                    epoch,
                    verbose=False
                    ):

        if verbose:
            print('Start of epoch %d' % (epoch,))
            print('Current alpha: %f' % (self.alpha,))
            print('Current resolution: {} * {}'.format(resolution, resolution))

        start = time.time()

        for step, (real_image_batch) in enumerate(dataset):
            self.checkpoint.step.assign_add(1)
            self.overall_steps += 1

            noise = tf.random.normal((self.resolution_batch_size, self.latent_dim))
            # self.discriminator_train_steps[str(resolution)](real_image_batch, noise, verbose=verbose)
            # self.generator_train_steps[str(resolution)](noise, verbose=verbose)

            self.discriminator_train_step(real_image_batch, noise, verbose=verbose)
            self.generator_train_step(noise, verbose=verbose)
            
            # update alpha
            if resolution > 4:
                self.alpha = min(1., self.alpha + self.resolution_alpha_increment)

            if real_image_batch.shape[0] < self.resolution_batch_size:
                raise ValueError('Image batch shape less than resolution batch size')

            # Write logged losses
            if self.overall_steps % self.loss_iter_evaluation == 0:
                
                # Log alpha
                self.metrics['alpha'](self.alpha)

                with self.gen_summary_writer.as_default():
                    tf.summary.scalar('generator_wasserstein_loss', self.metrics['generator_wasserstein_loss'].result(), step=self.overall_steps)
                    tf.summary.scalar('generator_loss', self.metrics['generator_loss'].result(), step=self.overall_steps)
                    tf.summary.scalar('alpha', self.metrics['alpha'].result(), step=self.overall_steps)

                with self.dis_summary_writer.as_default():
                    tf.summary.scalar('discriminator_wasserstein_loss_real', self.metrics['discriminator_wasserstein_loss_real'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_wasserstein_loss_fake', self.metrics['discriminator_wasserstein_loss_fake'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_wasserstein_gradient_penalty', self.metrics['discriminator_wasserstein_gradient_penalty'].result(), step=self.overall_steps)
                    # tf.summary.scalar('discriminator_epsilon_loss', self.metrics['discriminator_epsilon_loss'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_loss', self.metrics['discriminator_loss'].result(), step=self.overall_steps)

                # Save Images
                predicted_image = self.model.Generator(noise, training=False)
                predicted_image = predicted_image[:, :, :, :]* 0.5 + 0.5
                with self.gen_summary_writer.as_default():
                    tf.summary.image('Generated Images', predicted_image, max_outputs=16, step=self.overall_steps)

                # Take a look at real images
                real_image_batch = real_image_batch[:, :, :, :]* 0.5 + 0.5
                with self.dis_summary_writer.as_default():
                    tf.summary.image('Real Images', real_image_batch, max_outputs=5, step=self.overall_steps)

                sample_noise = tf.random.normal((9, self.latent_dim))
                self.generate_and_save_images(epoch, sample_noise, figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)

            # Save Checkpoint
            if self.overall_steps % self.save_iter == 0:
                self.save_check_point(resolution, verbose=True, save_to_gdrive=self.colab, g_drive_path = self.g_drive_path)

            # Reset Losses
            for k in self.metrics:
                self.metrics[k].reset_states()

        self.checkpoint.epoch.assign(epoch)

        print(f'Time for epoch {epoch} is {time.time()-start:.3f} sec. Training time: {time.time()-self.train_start_time:.3f}')

        if verbose:
            print('Completed')

        return True

    # @tf.function
    def discriminator_train_step(self, real_images, noise, verbose=False):
        epsilon = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0, maxval=1)

        with tf.GradientTape(persistent=True) as d_tape:
            with tf.GradientTape() as gp_tape:
                generated_images = self.model.Generator(noise, training=True)
                generated_images_mixed = epsilon * tf.dtypes.cast(real_images, tf.float32) + ((1 - epsilon) * generated_images)
                fake_mixed_pred = self.model.Discriminator(generated_images_mixed, training=True)
                
            # Compute gradient penalty
            grads = gp_tape.gradient(fake_mixed_pred, generated_images_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            discriminator_gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            if verbose:
                print('Obtained Wasserstein Gradient Penalty for Discriminator')
            
            discriminator_wloss_real = -tf.math.reduce_mean(self.model.Discriminator(real_images, training=True))
            if verbose:
                print('Obtained Wasserstein Loss for Discriminator on REAL images')
            discriminator_wloss_fake = tf.math.reduce_mean(self.model.Discriminator(generated_images, training=True))
            if verbose:
                print('Obtained Wasserstein Loss for Discriminator on FAKE images')

            total_discriminator_loss = discriminator_wloss_real + discriminator_wloss_fake + self.config.lambdaGP*discriminator_gradient_penalty

            # Log losses
            self.metrics['discriminator_wasserstein_loss_real'](discriminator_wloss_real)
            self.metrics['discriminator_wasserstein_loss_fake'](discriminator_wloss_fake)
            self.metrics['discriminator_wasserstein_gradient_penalty'](discriminator_gradient_penalty)
            self.metrics['discriminator_loss'](total_discriminator_loss)

        # Calculate the gradients for discriminator
        gradients_of_discriminator = d_tape.gradient(total_discriminator_loss, self.model.Discriminator.trainable_variables)
        if verbose:
            print('Computed discriminator loss gradients')

        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.Discriminator.trainable_variables))
        if verbose:
            print('Applied discriminator loss gradients')

    # @tf.function
    def generator_train_step(self, noise, verbose=False, return_generated_images=False):

        with tf.GradientTape() as g_tape:
            generated_images = self.model.Generator(noise, training=True)
            fake_predictions = self.model.Discriminator(generated_images, training=True)

            generator_wloss_fake = -tf.math.reduce_mean(fake_predictions)
            if verbose:
                print('Obtained Wasserstein Loss for Generator on FAKE images')

            total_generator_loss = generator_wloss_fake

            self.metrics['generator_wasserstein_loss'](generator_wloss_fake)
            self.metrics['generator_loss'](total_generator_loss)
        
        # Calculate the gradients for discriminator
        gradients_of_generator = g_tape.gradient(total_generator_loss, self.model.Generator.trainable_variables)
        if verbose:
            print('Computed generator loss gradients')
        # Apply the gradients to the optimizer
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.Generator.trainable_variables))
        if verbose:
            print('Applied generator loss gradients')

        if return_generated_images:
            return generated_images

    # @tf.function
    # def train_step(self, real_images, noise, return_generated_images=False, verbose=False):

    #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
    #         # 1. Real Output + Wasserstein Loss
    #         real_predictions = self.model.netD(real_images, training=True)
    #         discriminator_wloss_real = self.model.loss_criterion(real_predictions, True)
    #         if verbose:
    #             print('Obtained Wasserstein Loss for Discriminator on REAL images')

    #         # 2. Fake Output + Wasserstein Loss
    #         generated_images = self.model.netG(noise, training=True)
    #         fake_predictions = self.model.netD(generated_images, training=True)
    #         discriminator_wloss_fake = self.model.loss_criterion(fake_predictions, False)
    #         if verbose:
    #             print('Obtained Wasserstein Loss for Discriminator on FAKE images')
            
    #         generator_wloss_fake = self.model.loss_criterion(fake_predictions, True)
    #         if verbose:
    #             print('Obtained Wasserstein Loss for Generator on FAKE images')
            
    #         # 3. Wasserstein Gradient Penalty Loss
    #         if self.modelConfig.lambdaGP > 0:
    #             discriminator_gradient_penalty = WGANGPGradientPenalty(real_images, generated_images, self.model.netD, self.modelConfig.lambdaGP)
    #             if verbose:
    #                 print('Obtained Wasserstein Gradient Penalty for Discriminator')

    #         # 4. Epsilon Loss
    #         if self.modelConfig.epsilonD > 0:
    #             discriminator_episilon_loss = tf.math.reduce_mean(real_predictions[:,0]**2) + self.modelConfig.epsilonD
    #             if verbose:
    #                 print('Obtained Epsilon Loss for Discriminator')
    #         # total_discriminator_loss = discriminator_loss(real_predictions, fake_predictions)
    #         # total_generator_loss = generator_loss(fake_predictions)

    #         total_generator_loss = generator_wloss_fake
    #         total_discriminator_loss = discriminator_wloss_real + discriminator_wloss_fake + discriminator_episilon_loss + discriminator_gradient_penalty

    #         # Log losses
    #         self.metrics['discriminator_wasserstein_loss_real'](discriminator_wloss_real)
    #         self.metrics['discriminator_wasserstein_loss_fake'](discriminator_wloss_fake)
    #         self.metrics['discriminator_wasserstein_gradient_penalty'](discriminator_gradient_penalty)
    #         self.metrics['discriminator_epsilon_loss'](discriminator_episilon_loss)
    #         self.metrics['discriminator_loss'](total_discriminator_loss)

    #         self.metrics['generator_wasserstein_loss'](generator_wloss_fake)
    #         self.metrics['generator_loss'](total_generator_loss)

    #     gradients_of_generator = gen_tape.gradient(total_generator_loss, self.model.netG.trainable_variables)
    #     gradients_of_discriminator = disc_tape.gradient(total_discriminator_loss, self.model.netD.trainable_variables)
        
    #     if verbose:
    #         print('Computed loss gradients')
    #     self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.netG.trainable_variables))
    #     self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.netD.trainable_variables))
        
    #     if verbose:
    #         print('Applied loss gradients')

    #     if return_generated_images:
    #         return generated_images
