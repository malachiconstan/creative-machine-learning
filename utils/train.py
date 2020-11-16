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

        self.model.load_weights(self.checkpoint_dir)
        file_paths = glob(os.path.join(infer_datadir,'*.jpeg'))
        test_pred = tf.stack([process_path(file,img_height,img_width,False,False) for file in file_paths])

        preds = tf.nn.softmax(self.model(test_pred),axis=1).numpy()
        df_preds = pd.DataFrame(preds)
        df_preds.index = os.listdir(infer_datadir)
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
    _defaultConfig = _C

    def getDefaultConfig(self):
        return ProgressiveGANTrainer._defaultConfig

    def __init__(self,
                 datapath,
                 discriminator_optimizer,
                 generator_optimizer,
                 loss_iter_evaluation=200,
                 save_iter=5000,
                 model_label="PGGAN",
                 config=None
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

        self.datapath = datapath
        self.log_dir = os.path.join(os.getcwd(),'pggan_logs')
        self.gen_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'gen')
        self.dis_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'dis')
        self.checkpoint_dir = os.path.join(os.getcwd(),'pggan_checkpoints')
        self.train_config_path = os.path.join(self.checkpoint_dir, f'{model_label}_' + "_train_config.json")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if config is None:
            config = {}

        self.readTrainConfig(config)

        # Intern state
        self.startScale = 0
        self.startIter = 0
        self.overall_steps = 0

        self.initModel()

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        # Checkpoints
        self.model_label = model_label
        self.save_iter = save_iter
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.model.netG,
            discriminator=self.model.netD
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # Logging
        self.loss_iter_evaluation = loss_iter_evaluation
        self.metrics = dict(
            discriminator_wasserstein_loss_real = tf.keras.metrics.Mean('discriminator_wasserstein_loss_real', dtype=tf.float32),
            discriminator_wasserstein_loss_fake = tf.keras.metrics.Mean('discriminator_wasserstein_loss_fake', dtype=tf.float32),
            discriminator_wasserstein_gradient_penalty = tf.keras.metrics.Mean('discriminator_wasserstein_gradient_penalty', dtype=tf.float32),
            generator_wasserstein_loss = tf.keras.metrics.Mean('generator_wasserstein_loss', dtype=tf.float32),
            discriminator_epsilon_loss = tf.keras.metrics.Mean('discriminator_epsilon_loss', dtype=tf.float32),
            discriminator_loss = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32),
            generator_loss = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32),
        )

        self.gen_summary_writer = tf.summary.create_file_writer(self.gen_log_dir)
        self.dis_summary_writer = tf.summary.create_file_writer(self.dis_log_dir)

        # Low-res layers
        self.avgpool2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
        self.upsampling2d = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')        
        
    def initModel(self, **kwargs):
        """
        Initialize the GAN model.
        """

        self.model = ProgressiveGAN(
            latent_dim = self.modelConfig.latent_dim,
            level_0_channels = self.modelConfig.depthScales[0],
            init_bias_zero = self.modelConfig.init_bias_zero,
            leaky_relu_leak = self.modelConfig.leaky_relu_leak,
            per_channel_normalisation = self.modelConfig.per_channel_normalisation,
            mini_batch_sd = self.modelConfig.mini_batch_sd,
            equalizedlR = self.modelConfig.equalizedlR,
            output_dim = self.modelConfig.output_dim,
            GDPP = self.modelConfig.GDPP,
            lambdaGP = self.modelConfig.lambdaGP,
            **kwargs
        )

    def readTrainConfig(self, config, verbose=True):
        """
        Load a permanent configuration describing a models. The variables
        described in this file are constant through the training.
        """
        self.modelConfig = BaseConfig()
        getConfigFromDict(self.modelConfig, config, self.getDefaultConfig())

        if self.modelConfig.alphaJumpMode not in ["custom", "linear"]:
            raise ValueError(
                "alphaJumpMode should be one of the followings: \
                'custom', 'linear'")

        if self.modelConfig.alphaJumpMode == "linear":

            self.modelConfig.alphaNJumps[0] = 0
            self.modelConfig.iterAlphaJump = []
            self.modelConfig.alphaJumpVals = []

            self.updateAlphaJumps(self.modelConfig.alphaNJumps, self.modelConfig.alphaSizeJumps)
            
            if verbose:
                print('Linear Alpha Jump Vals Updated')

        self.scaleSanityCheck()
        
        if verbose:
            print('Training Configuration Read')

    def scaleSanityCheck(self, verbose=True):
        '''
        Sanity check
        Makes sures that the lists of:
            * # Channel per scale
            * Maximum iteration per scale
            * Number of alpha jumps per scale
            * The iterations at which the alpha jumps at each scale
        are all of the same size.
        '''
        n_scales = min(len(self.modelConfig.depthScales),
                       len(self.modelConfig.maxIterAtScale),
                       len(self.modelConfig.iterAlphaJump),
                       len(self.modelConfig.alphaJumpVals))

        assert self.modelConfig.depthScales == self.modelConfig.depthScales[:n_scales], 'Size of depthScales wrong'
        assert self.modelConfig.maxIterAtScale == self.modelConfig.maxIterAtScale[:n_scales], 'Size of maximum iter/scale wrong'
        assert self.modelConfig.iterAlphaJump == self.modelConfig.iterAlphaJump[:n_scales], 'Size of iterations at which alpha jumps wrong'
        assert self.modelConfig.alphaJumpVals == self.modelConfig.alphaJumpVals[:n_scales], 'Size of alpha values per scale wrong'

        self.modelConfig.size_scales = [4]
        for _ in range(1, n_scales):
            self.modelConfig.size_scales.append(
                self.modelConfig.size_scales[-1] * 2)

        self.modelConfig.n_scales = n_scales

        if verbose:
            print('Scale Sanity Check Completed')
            print('Scales: ',n_scales)
            print('Scale Sizes: ',self.modelConfig.size_scales)

        # Training functions
        self.train_steps = [deepcopy(self.train_step) for i in range(n_scales)]

    def updateAlphaJumps(self, nJumpScale, sizeJumpScale):
        """
        Given the number of iterations between two updates of alpha at each
        scale and the number of updates per scale, build the effective values of
        self.maxIterAtScale and self.alphaJumpVals.
        Example: If the number of iterations between 2 jumps is 32, and alpha has to be updated 600 times then the 
        number of iterations at which alpha is updated will be 0, 32, 64 ... 19200 (600*32) and alpha will progressively fall from 1 to 0.

        Args:
            - nJumpScale (list of int): for each scale, the number of times
                                        alpha should be updated
            - sizeJumpScale (list of int): for each scale, the number of
                                           iterations between two updates
        """

        n_scales = min(len(nJumpScale), len(sizeJumpScale))

        for scale in range(n_scales):

            self.modelConfig.iterAlphaJump.append([])
            self.modelConfig.alphaJumpVals.append([])

            if nJumpScale[scale] == 0:
                self.modelConfig.iterAlphaJump[-1].append(0)
                self.modelConfig.alphaJumpVals[-1].append(0.0)
                continue

            diffJump = 1.0 / float(nJumpScale[scale])
            currVal = 1.0
            currIter = 0

            while currVal > 0:

                self.modelConfig.iterAlphaJump[-1].append(currIter)
                self.modelConfig.alphaJumpVals[-1].append(currVal)

                currIter += sizeJumpScale[scale]
                currVal -= diffJump

            self.modelConfig.iterAlphaJump[-1].append(currIter)
            self.modelConfig.alphaJumpVals[-1].append(0.0)

    def inScaleUpdate(self, iter, scale, input_real):

        if self.indexJumpAlpha < len(self.modelConfig.iterAlphaJump[scale]):
            if iter == self.modelConfig.iterAlphaJump[scale][self.indexJumpAlpha]:
                alpha = self.modelConfig.alphaJumpVals[scale][self.indexJumpAlpha]
                self.model.updateAlpha(alpha)
                self.indexJumpAlpha += 1

        if self.model.config.alpha > 0:
            low_res_real = self.avgpool2d(input_real)
            low_res_real = self.upsampling2d(low_res_real)

            alpha = self.model.config.alpha
            input_real = alpha * low_res_real + (1-alpha) * input_real

        return input_real

    def addNewScales(self, configNewScales):

        if configNewScales["alphaJumpMode"] not in ["custom", "linear"]:
            raise ValueError("alphaJumpMode should be one of the followings: \
                            'custom', 'linear'")

        if configNewScales["alphaJumpMode"] == 'custom':
            self.modelConfig.iterAlphaJump = self.modelConfig.iterAlphaJump + \
                configNewScales["iterAlphaJump"]
            self.modelConfig.alphaJumpVals = self.modelConfig.alphaJumpVals + \
                configNewScales["alphaJumpVals"]

        else:
            self.updateAlphaJumps(configNewScales["alphaNJumps"],
                                  configNewScales["alphaSizeJumps"])

        self.modelConfig.depthScales = self.modelConfig.depthScales + \
            configNewScales["depthScales"]
        self.modelConfig.maxIterAtScale = self.modelConfig.maxIterAtScale + \
            configNewScales["maxIterAtScale"]

        self.scaleSanityCheck()
    
    def saveBaseConfig(self):
        """
        Save the model basic configuration (the part that doesn't change with
        the training's progression) at the given path
        """

        outConfig = getDictFromConfig(self.modelConfig, self.getDefaultConfig())

        if "alphaJumpMode" in outConfig:
            if outConfig["alphaJumpMode"] == "linear":

                outConfig.pop("iterAlphaJump", None)
                outConfig.pop("alphaJumpVals", None)

        with open(self.train_config_path, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

    def save_check_point(self, scale, iter, verbose=True, save_to_gdrive=True, g_drive_path = '/content/drive/My Drive/CML'):
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

        # Tmp Configuration
        outConfig = {'scale': scale, 'iter': iter}

        with open(self.temp_config_path, 'w') as fp:
            json.dump(outConfig, fp, indent=4)
        
        if verbose:
            print('Saved temp outconfig to: ',self.temp_config_path)

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

        # Load the temp configuration
        # Find latest scale file
        scale = 0
        for scale in range(self.modelConfig.n_scales-1,-1,-1):
            print(scale)
            path = os.path.join(self.checkpoint_dir, f'{self.model_label}_{scale}_' + "_tmp_config.json")
            print(path)
            if os.path.exists(path):
                self.temp_config_path = path
                break

        with open(self.temp_config_path,'rb') as infile:
            tmpConfig = json.load(infile)
        self.startScale = tmpConfig["scale"]
        self.startIter = tmpConfig["iter"]

        # Read the training configuration
        with open(self.train_config_path,'rb') as infile:
            trainConfig = json.load(infile)
        self.readTrainConfig(trainConfig)

        # Re-initialize the model
        self.initModel(depthOtherScales = [self.modelConfig.depthScales[i] for i in range(0, self.startScale)])
        
        # Load saved checkpoint
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")

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

        if self.log_dir is not None:
            self.saveBaseConfig()

        for scale in range(self.startScale, self.modelConfig.n_scales):
            print(f'Scale {scale} for size {self.modelConfig.size_scales[scale]} training begins')

            # Define specific paths
            self.temp_config_path = os.path.join(self.checkpoint_dir, f'{self.model_label}_{scale}_' + "_tmp_config.json")
            
            # Get train dataset at the correct image scale
            train_dataset = get_image_dataset(self.datapath,
                                            img_height=self.modelConfig.size_scales[scale],
                                            img_width=self.modelConfig.size_scales[scale],
                                            batch_size=self.modelConfig.miniBatchSize,
                                            normalize=True)
            
            if verbose:
                print(f'Dataset for size {self.modelConfig.size_scales[scale]} obtained')

            self.step = 0
            if self.startIter > 0:
                self.step = self.startIter
                self.overall_steps = self.startIter + np.sum([self.modelConfig.maxIterAtScale[i] for i in range(0, self.startScale)])
                self.startIter = 0

            shiftAlpha = 0

            # While the shiftAlpha variable is less than the jumps of alpha in that scale and the iteration which the shiftAlpha corresponds to in that scale is less than the shiftIter, add 1 to shiftAlpha
            # Basically this tells us what is the level of alpha we should start at (the one right before the shiftIter)
            while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and self.modelConfig.iterAlphaJump[scale][shiftAlpha] < self.step:
                shiftAlpha += 1

            while self.step < self.modelConfig.maxIterAtScale[scale]:
                
                # Set the index to set alpha to to the current shiftAlpha
                self.indexJumpAlpha = shiftAlpha
                
                status = self.train_epoch(train_dataset, scale, maxIter=self.modelConfig.maxIterAtScale[scale], verbose=verbose)

                if not status:
                    return False
                
                # Update shiftAlpha to the next step
                while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and self.modelConfig.iterAlphaJump[scale][shiftAlpha] < self.step:
                    shiftAlpha += 1

            # If final scale then don't add anymore layers
            if scale == self.modelConfig.n_scales - 1:
                break

            # Add scale
            self.model.addScale(self.modelConfig.depthScales[scale + 1])

        self.startScale = self.modelConfig.n_scales
        self.startIter = self.modelConfig.maxIterAtScale[-1]
        return True


    def train_epoch(self,
                    dataset,
                    scale,
                    maxIter=-1,
                    verbose=True
                    ):
        """
        Train the model on one epoch.
        Args:
            - dbLoader (DataLoader): dataset on which the training will be made
            - scale (int): scale at which is the training is performed
            - shiftIter (int): shift to apply to the iteration index when
                               looking for the next update of the alpha
                               coefficient
            - maxIter (int): if > 0, iteration at which the training should stop
        Returns:
            True if the training went smoothly
            False if a diverging behavior was detected and the training had to
            be stopped
        """

        if verbose:
            print('Dataset Length: ', len(dataset))

        start = time.time()
        previous_step = self.step, self.overall_steps

        for real_image_batch in dataset:
            self.step += 1
            self.overall_steps += 1
            self.checkpoint.step.assign_add(1)

            if real_image_batch.shape[0] < self.modelConfig.miniBatchSize:
                raise ValueError('Image batch shape less than mini_batch_size')

            # Additional updates inside a scale
            real_image_batch = self.inScaleUpdate(self.step, scale, real_image_batch)
            noise = tf.random.normal([real_image_batch.shape[0], self.modelConfig.latent_dim])
            
            self.train_steps[scale](real_images=real_image_batch, noise=noise, verbose=verbose)

            # Write logged losses
            if self.overall_steps % self.loss_iter_evaluation == 0:

                with self.gen_summary_writer.as_default():
                    tf.summary.scalar('generator_wasserstein_loss', self.metrics['generator_wasserstein_loss'].result(), step=self.overall_steps)
                    tf.summary.scalar('generator_loss', self.metrics['generator_loss'].result(), step=self.overall_steps)

                with self.dis_summary_writer.as_default():
                    tf.summary.scalar('discriminator_wasserstein_loss_real', self.metrics['discriminator_wasserstein_loss_real'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_wasserstein_loss_fake', self.metrics['discriminator_wasserstein_loss_fake'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_wasserstein_gradient_penalty', self.metrics['discriminator_wasserstein_gradient_penalty'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_epsilon_loss', self.metrics['discriminator_epsilon_loss'].result(), step=self.overall_steps)
                    tf.summary.scalar('discriminator_loss', self.metrics['discriminator_loss'].result(), step=self.overall_steps)

                # Save Images
                predicted_image = self.model.netG(noise, training=False)
                predicted_image = predicted_image[:, :, :, :]* 0.5 + 0.5
                with self.gen_summary_writer.as_default():
                    tf.summary.image('Generated Images', predicted_image, max_outputs=16, step=self.overall_steps)

                # Take a look at real images
                real_image_batch = real_image_batch[:, :, :, :]* 0.5 + 0.5
                with self.dis_summary_writer.as_default():
                    tf.summary.image('Real Images', real_image_batch, max_outputs=5, step=self.overall_steps)

            # Save Checkpoint
            if self.overall_steps % self.save_iter == 0:
                self.save_check_point(scale, self.step, verbose=True, save_to_gdrive=self.colab, g_drive_path = self.g_drive_path)

            # Reset Losses
            for k in self.metrics:
                self.metrics[k].reset_states()
            
            if self.step == maxIter:
                if verbose:
                    print('Max iterations reached')
                return True

        print(f'Time from step {previous_step[0]}/{previous_step[1]} to {self.step}/{maxIter}, {self.overall_steps} is {time.time()-start:.3f} sec. Training time: {time.time()-self.train_start_time:.3f}')

        if verbose:
            print('Completed')

        return True
    
    @tf.function
    def train_step(self, real_images, noise, return_generated_images=False, verbose=False):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # 1. Real Output + Wasserstein Loss
            real_predictions = self.model.netD(real_images, training=True)
            discriminator_wloss_real = self.model.loss_criterion(real_predictions, True)
            if verbose:
                print('Obtained Wasserstein Loss for Discriminator on REAL images')

            # 2. Fake Output + Wasserstein Loss
            generated_images = self.model.netG(noise, training=True)
            fake_predictions = self.model.netD(generated_images, training=True)
            discriminator_wloss_fake = self.model.loss_criterion(fake_predictions, False)
            if verbose:
                print('Obtained Wasserstein Loss for Discriminator on FAKE images')
            
            generator_wloss_fake = self.model.loss_criterion(fake_predictions, True)
            if verbose:
                print('Obtained Wasserstein Loss for Generator on FAKE images')
            
            # 3. Wasserstein Gradient Penalty Loss
            if self.modelConfig.lambdaGP > 0:
                discriminator_gradient_penalty = WGANGPGradientPenalty(real_images, generated_images, self.model.netD, self.modelConfig.lambdaGP)
                if verbose:
                    print('Obtained Wasserstein Gradient Penalty for Discriminator')

            # 4. Epsilon Loss
            if self.modelConfig.epsilonD > 0:
                discriminator_episilon_loss = tf.math.reduce_mean(real_predictions[:,0]**2) + self.modelConfig.epsilonD
                if verbose:
                    print('Obtained Epsilon Loss for Discriminator')
            # total_discriminator_loss = discriminator_loss(real_predictions, fake_predictions)
            # total_generator_loss = generator_loss(fake_predictions)

            total_generator_loss = generator_wloss_fake
            total_discriminator_loss = discriminator_wloss_real + discriminator_wloss_fake + discriminator_episilon_loss + discriminator_gradient_penalty

            # Log losses
            self.metrics['discriminator_wasserstein_loss_real'](discriminator_wloss_real)
            self.metrics['discriminator_wasserstein_loss_fake'](discriminator_wloss_fake)
            self.metrics['discriminator_wasserstein_gradient_penalty'](discriminator_gradient_penalty)
            self.metrics['discriminator_epsilon_loss'](discriminator_episilon_loss)
            self.metrics['discriminator_loss'](total_discriminator_loss)

            self.metrics['generator_wasserstein_loss'](generator_wloss_fake)
            self.metrics['generator_loss'](total_generator_loss)

        gradients_of_generator = gen_tape.gradient(total_generator_loss, self.model.netG.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(total_discriminator_loss, self.model.netD.trainable_variables)
        
        if verbose:
            print('Computed loss gradients')
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.netG.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.netD.trainable_variables))
        
        if verbose:
            print('Applied loss gradients')

        if return_generated_images:
            return generated_images
