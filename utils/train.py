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
from copy import deepcopy, copy
from glob import glob

from IPython.display import clear_output
from utils.models import ProgressiveGAN
from utils.preprocessing import get_image_dataset, process_path
from utils.losses import cggan_discriminator_loss, cggan_generator_loss, identity_loss, cycle_loss

# TODO: Create DCGAN Trainer Class

def discriminator_loss(real_output, fake_output):
    '''
    Return discriminator binary cross entropy loss. Takes loss using random.uniform reference, rather than int 1 or 0.
    
    :params:
        tf.Tensor real_output: Tensor output of discriminator using real images, in shape of [b, 1]
        tf.Tensor fake_output: Training dataset of discriminator using fake images, in shape of [b, 1]

    :return:
        tf.Tensor total_loss: Loss, in shape [b, 1]
    '''
    real_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([real_output.shape[0],1],0.7,1.2), real_output, from_logits=True) # set noise to 1
    fake_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0,0.3), fake_output, from_logits=True) # set noise to 0
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    '''
    Return generator binary cross entropy loss. Takes loss using random.uniform reference, rather than int 1 or 0.
    
    :params:
        tf.Tensor fake_output: Training dataset of discriminator using fake images, in shape of [b, 1]

    :return:
        tf.Tensor total_loss: Loss, in shape [b, 1]
    '''
    return tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0.7,1.2), fake_output, from_logits=True) # set noise to 1

@tf.function
def train_step(
        images,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        latent_dim,
        batch_size,
        sdis_loss,
        sgen_loss,
        sdis_acc
    ):
    '''
    Train step for DCGAN. Get loss from generator and discriminator, then apply back propagation to optimise generator and discriminator weights.
    Function is annotated as tf.function, cannot change generator or discriminator once function is defined
    
    :params:
        tf.Tensor images: Tensor of images in shape [b, h, w, c]
        tf.keras.Model generator: Generator Model
        tf.keras.Model discriminator: Discriminator Model
        tf.keras.Optimizer generator_optimizer: Generator Optimizer
        tf.keras.Optimizer discriminator_optimizer: Discriminator Optimizer
        int latent_dim: Latent Dimension of Generator e.g. 512
        int batch_size: Batch Size
        tf.keras.metrics.Mean sdis_loss: Discriminator Loss
        tf.keras.metrics.Mean sgen_loss: Generator Loss
        tf.keras.metrics.Mean sdis_acc: Discriminator Accuracy
    '''
    # Define Latent Noise Tensor
    noise = tf.random.normal([batch_size, latent_dim])

    # Record Gradients of Discriminator and Generator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)

        # Get discriminator output of real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Compute losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Record losses and accuracy
        sdis_loss(disc_loss)
        sgen_loss(gen_loss)
        sdis_acc(tf.ones_like(real_output), real_output)

    # Obtain Discriminator and Generator Gradients using backpropagation
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply Gradient Descent
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input, file_writer):
    '''
    Generate and save images to tensorboard
    
    :params:
        tf.keras.Model model: Generator Model
        int epoch: Epoch at which model is trained
        tf.Tensor test_input: Latent Noise Tensor
        tf.summary.FileWriter file_writer: Tensorboard file writer
    '''
    
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = predictions[:, :, :, :]* 0.5 + 0.5

    # Save Images to Tensorboard
    with file_writer.as_default():
        tf.summary.image('Generated Images', predictions, max_outputs=16, step=epoch)

def train(
        dataset,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        epochs,
        batch_size,
        latent_dim,
        restore=False,
        save_step=100,
        saveimg_step=10
    ):
    '''
    Train DCGAN
    
    :params:
        tf.keras.Dataset dataset: Keras Dataset of Images in shape [b, h, w, c]
        tf.keras.Model generator: Generator Model
        tf.keras.Model discriminator: Discriminator Model
        tf.keras.Optimizer generator_optimizer: Generator Optimizer
        tf.keras.Optimizer discriminator_optimizer: Discriminator Optimizer
        int epochs: Total epochs to train
        int latent_dim: Latent Dimension of Generator e.g. 512
        int batch_size: Batch Size
        bool restore: Whether to restore from saved checkpoint
        int save_step: Save checkpoint every save_step steps
        int saveimg_step: Output and save image every saveimg_step steps
    '''
    
    # Generate seed to generate images later
    seed = tf.random.normal([16,latent_dim])

    # Define directories
    log_dir = os.path.join(os.getcwd(), 'logs')
    output_dir = os.path.join(os.getcwd(), 'outputs')
    checkpoint_path = os.path.join(os.getcwd(),'checkpoints')
    
    # Define log directories
    current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")
    gen_log_dir = os.path.join(log_dir,'gradient_tape',current_time,'gen')
    dis_log_dir = os.path.join(log_dir,'gradient_tape',current_time,'dis')

    # Create directories if directories do not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Get checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    # Define metrics
    sgen_loss = tf.keras.metrics.Mean('sgen_loss', dtype=tf.float32)
    sdis_loss = tf.keras.metrics.Mean('sdis_loss', dtype=tf.float32)
    sdis_acc = tf.keras.metrics.BinaryAccuracy('sdis_acc')
    
    # Define file writers
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

    # Epoch training step
    for epoch in range(epochs):

        if restore:
            step=int(checkpoint.step)+epoch
        else:
            step=epoch

        start = time.time()

        # Train step
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, latent_dim, batch_size, sdis_loss, sgen_loss, sdis_acc)

        # Write losses and accuracy
        with gen_summary_writer.as_default():
            tf.summary.scalar('sgen_loss', sgen_loss.result(), step=step)

        with dis_summary_writer.as_default():
            tf.summary.scalar('sdis_loss', sdis_loss.result(), step=step)
            tf.summary.scalar('sdis_acc', sdis_acc.result(), step=step)

        # generate images
        display.clear_output(wait=True)
        if (epoch + 1 + add_step)%saveimg_step==0:
            generate_and_save_images(generator,epoch,seed,gen_summary_writer)

        # Save checkpoint
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
        
        # Reset metrics
        sgen_loss.reset_states()
        sgen_loss.reset_states()
        sdis_loss.reset_states()

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epoch,seed,gen_summary_writer)

class ClassifierTrainer(object):
    '''
    Trainer class for classifier. Initialise this class, then call train() method to train the classifier.
    The class takes care of logging and model saving, as well as learning rate scheduling.
    '''
    def __init__(self,
                train_dataset,
                validation_dataset,
                model,
                optimizer,
                lr_schedule
                ):
        '''
        __init__ method. Instantiates logs and checkpoint directories, as well as relevant callbacks to be used for training

        :params:
        tf.keras.Dataset train_dataset: Training dataset
        tf.keras.Dataset validataion_dataset: Training dataset
        tf.keras.Model model: Classifier Model. Typically one pre-trained on Imagenet
        tf.keras.Optimizer optimizer: Optimizer used for training e.g. Adam
        function lr_schedule: A function that takes in an epoch and returns the relevant learning rate
        '''

        # Define Directory paths
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")
        self.__log_dir = os.path.join(os.getcwd(),'classifier_logs',current_time)
        self.__checkpoint_dir = os.path.join(os.getcwd(),'classifier_checkpoints')

        # Create Directories if directories do not exist
        if not os.path.exists(self.__log_dir):
            os.makedirs(self.__log_dir)

        if not os.path.exists(self.__checkpoint_dir):
            os.makedirs(self.__checkpoint_dir)

        # Define tensorboard callback to track loss and accuracy on tensorboard
        self.__tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.__log_dir)

        # Define Checkpoint callback to save model every epoch
        self.__cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.__checkpoint_dir,'cp.ckpt'),
                                                                verbose=1,
                                                                save_weights_only=True,
                                                                save_freq = 'epoch')

        # Store Model and Optimizer
        self.__model = model
        self.__optimizer = optimizer

        # Store train and validation datasets
        self.__train_dataset = train_dataset
        self.__validation_dataset = validation_dataset

        # Define Learning Rate Scheduler and log learning rate schedule to 
        self.__file_writer = tf.summary.create_file_writer(self.__log_dir + "/metrics")
        self.__file_writer.set_as_default()

        # Set learning rate callback to decrease learning rate during training
        self.__lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # Compile Model with Loss and Optimizer
        self.__model.compile(
            optimizer = self.__optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ["accuracy"]
        )
    
    def train(self,
            epochs=100,
            batch_size=32
            ):
        '''
        train method. Trains the model using the stored train and validation datasets, then returns the history

        :params:
        int epochs: total epochs to train for
        int batch_size: Batch Size
        '''

        self.history = self.__model.fit(self.__train_dataset,
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[self.__cp_callback, self.__tensorboard_callback, self.__lr_callback],
                                validation_data=self.__validation_dataset)

        print('Training Completed')

    def infer(self,
            infer_datadir,
            img_height,
            img_width
            ):
        '''
        infer method. Generate an excel sheet showing the percentage allocated to each year for each inferred image.
        Inferred images should be stored in infer_datadir and all be in either jpg or jpeg format.
        img_width and img_height should be the same as that of the model.

        :params:
        str infer_datadir: str or os.path that stores a path to the infer_datadir
        int img_height: Image height to rescale to
        int img_width: Image Width to rescale to
        '''

        # Load the weights in the checkpoint
        self.__model.load_weights(os.path.join(self.__checkpoint_dir,'cp.ckpt'))

        # Get all file paths of all images in the infer data directory using glob
        file_paths = glob(os.path.join(infer_datadir,'*.jpeg')) + glob(os.path.join(infer_datadir,'*.jpg'))

        # Create a tensor for all the infer images. Use the process_path function to convert an image to a tensor
        test_pred = tf.stack([process_path(file,img_height,img_width,False,False) for file in file_paths])

        # Apply softmax after predicting using the model to get the class probabilities
        preds = tf.nn.softmax(self.__model(test_pred),axis=1).numpy()

        # Create a new dataframe from the class probabilities array
        df_preds = pd.DataFrame(preds)

        # The row names are the file names
        df_preds.index = [os.path.split(fp)[1] for fp in file_paths]

        # Column names are the class names
        df_preds.columns = self.__train_dataset.class_names
        
        # Save predictions to a csv file
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
    '''
    PGGAN Trainer Class. Initialise this class, then call train method
    '''

    def __init__(
            self,
            config,
            discriminator_optimizer,
            generator_optimizer
        ):
        '''
        Constructor.

        :params:
            edict config: easy dict with key stored as attributes
            tf.keras.Optimizer discriminator_optimizer: Discriminator Optimizer
            tf.keras.Optimizer generator_optimizer: Generator Optimizer
        '''

        # Define directories
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")

        self.datapath = config.datapath
        self.log_dir = os.path.join(os.getcwd(),'pggan_logs')
        self.img_dir = os.path.join(os.getcwd(),'pggan_imgs')
        self.gen_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'gen')
        self.dis_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'dis')
        self.checkpoint_dir = os.path.join(os.getcwd(),'pggan_checkpoints')
        self.model_save_dir = os.path.join(os.getcwd(),'pggan_weights')
        self.train_config_path = os.path.join(self.checkpoint_dir, f'{config.model_label}_' + "_train_config.json")

        # Create directories if non existent
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Store Config
        self.config = config

        # Store Config Parameters in This object
        self.start_resolution = config.resolution
        self.stop_resolution = config.stop_resolution
        self.start_epoch = config.start_epoch
        self.epochs = config.epochs
        self.batch_size = self.calculate_batch_size(config.resolution)
        self.latent_dim = config.latent_dim
        self.overall_steps = 0
        self.hard_start = config.hard_start

        # Create Model
        self.__initialise_model()

        # Store Generators
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        # Define Checkpoints
        self.model_label = config.model_label
        self.save_iter = config.save_iter
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
            epoch = tf.Variable(self.start_epoch),
            resolution = tf.Variable(self.start_resolution),
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.model.Generator,
            discriminator=self.model.Discriminator
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # Define metrics
        self.loss_iter_evaluation = config.loss_iter_evaluation
        self.metrics = dict(
            discriminator_wasserstein_loss_real = tf.keras.metrics.Mean('discriminator_wasserstein_loss_real', dtype=tf.float32),
            discriminator_wasserstein_loss_fake = tf.keras.metrics.Mean('discriminator_wasserstein_loss_fake', dtype=tf.float32),
            discriminator_wasserstein_gradient_penalty = tf.keras.metrics.Mean('discriminator_wasserstein_gradient_penalty', dtype=tf.float32),
            generator_wasserstein_loss = tf.keras.metrics.Mean('generator_wasserstein_loss', dtype=tf.float32),
            discriminator_loss = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32),
            generator_loss = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32),
            alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)
        )

        # Create summary writers
        self.gen_summary_writer = tf.summary.create_file_writer(self.gen_log_dir)
        self.dis_summary_writer = tf.summary.create_file_writer(self.dis_log_dir)

        # Create training steps dict to store tf.function training steps
        self.discriminator_train_steps = dict()
        self.generator_train_steps = dict()

    def __initialise_model(
            self,
            resolution=None
        ):
        '''
        Private method
        Create and store PGGAN model

        :params:
            int resolution: Resolution of PGGAN to create. Choose from 4, 8, 16, 32, 64, 128, 256, 512
        '''

        # Set resolution to start resolution if no resolution defined
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

    def __generate_and_save_images(
            self,
            epoch,
            figure_size=(12,6),
            subplot=(3,6),
            save=True
        ):
        '''
        Private method
        Generate and save images using the Generator

        :params:
            int epoch: Current epoch
            tuple figure_size: Size of figure for matplotlib.pyplot
            tuple subplot: number of subplots to prepare
            bool save: Whether to save the subplots
        '''

        # Create sample noise tensor
        sample_noise = tf.random.normal((9, self.latent_dim), seed=0)

        # Create alpha tensor
        alpha_tensor = tf.constant(np.repeat(self.alpha, 9).reshape(9, 1), dtype=tf.float32)

        # Generate images and save into subplots
        generated_images = self.model.Generator((sample_noise, alpha_tensor))
        fig = plt.figure(figsize=figure_size)
        for i in range(generated_images.shape[0]):
            axs = plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(generated_images[i] * 0.5 + 0.5)
            plt.axis('off')
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.img_dir, '{}x{}_image_at_epoch_{:04d}.png'.format(generated_images.shape[1], generated_images.shape[2], epoch)))
        
        # Close plot
        plt.close()

    def __save_check_point(
            self,
            resolution,
            verbose=True,
            save_to_gdrive=True,
            g_drive_path = '/content/drive/My Drive/CML'
        ):
        '''
        Private method
        Save checkpoint and weights for PGGAN

        :params:
            int resolution: Current resolution
            bool verbose: Verbosity
            bool save_to_gdrive: Whether to save the checkpoints to google drive
            str g_drive_path: Path to save checkpoints in gdrive
        '''
        # Get checkpoint dir from checkpoint manager
        save_path = self.checkpoint_manager.save()
        if verbose:
            print('Checkpoint step at: ',int(self.checkpoint.step))
            print(f"Saved checkpoint for step {int(self.checkpoint.step)}: {save_path}")
        
        # Save weights as h5 files
        self.model.Generator.save_weights(os.path.join(self.model_save_dir, f'{resolution}x{resolution}_generator.h5'))
        self.model.Discriminator.save_weights(os.path.join(self.model_save_dir, f'{resolution}x{resolution}_discriminator.h5'))

        if save_to_gdrive:
            # Save the files to Google Drive for permanent storage
            from utils.drive_helper import copy_to_gdrive

            # Check if GDrive mounted, if yes, then make dir if dir does not exist
            if not os.path.exists(g_drive_path):
                if not os.path.exists('/content/drive/My Drive/'):
                    raise NotADirectoryError('Drive not mounted')
                os.makedirs(g_drive_path)

            # Define save paths in GDrive
            checkpoint_path = os.path.join(g_drive_path,'checkpoints.zip')
            weights_path = os.path.join(g_drive_path,'weights.zip')
            logs_path = os.path.join(g_drive_path,'logs.zip')

            # Copy from local to GDrive
            copy_to_gdrive(local_path=self.checkpoint_dir, g_drive_path=checkpoint_path)
            copy_to_gdrive(local_path=self.model_save_dir, g_drive_path=weights_path)
            copy_to_gdrive(local_path=self.log_dir, g_drive_path=logs_path)

            print('Checkpoints Saved to ',checkpoint_path)
            print('Weights Saved to ',weights_path)
            print('Logs Saved to ',logs_path)

    def __load_saved_training(
            self,
            load_from_g_drive=False,
            load_weights=True
        ):
        '''
        Private method
        Load checkpoints and weights

        :params:
            bool load_from_g_drive: Whether to load from GDrive
            bool load_weights: Whether to load weights
        '''
        
        # Extract checkpoints and weights from Google Drive if load_from_g_drive and using colab
        if self.colab and load_from_g_drive:
            from utils.drive_helper import extract_data_g_drive
            extract_data_g_drive('CML/checkpoints.zip', mounted=True, extracting_checkpoints=True)
            print('Extracted checkpoints from colab')
            extract_data_g_drive('CML/weights.zip', mounted=True, extracting_checkpoints=True, checkpoint_dir = 'pggan_weights')
            print('Extracted weights from colab')

        # Get resolution, current step and start epoch from latest checkpoint
        self.start_resolution = tf.train.load_variable(self.checkpoint_dir, 'resolution/.ATTRIBUTES/VARIABLE_VALUE')
        self.overall_steps = tf.train.load_variable(self.checkpoint_dir, 'step/.ATTRIBUTES/VARIABLE_VALUE')
        self.start_epoch = tf.train.load_variable(self.checkpoint_dir, 'epoch/.ATTRIBUTES/VARIABLE_VALUE')
        self.__initialise_model()

        # Load saved checkpoint
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")

        # Load Weights
        if load_weights:
            self.__load_weights(self.start_resolution)
        
        print(f'Start training from {self.start_resolution}x{self.start_resolution} at epoch: {self.start_epoch}, step: {self.overall_steps}')

    def __load_weights(
            self,
            load_resolution
        ):
        '''
        Private method
        Load weights

        :params:
            int load_resolution: Resolution to load weights from
        '''
        
        # Check if weights exist, then load weights
        generator_weights_path = os.path.join(self.model_save_dir, f'{load_resolution}x{load_resolution}_generator.h5')
        if os.path.isfile(generator_weights_path):
            self.model.Generator.load_weights(generator_weights_path, by_name=True)
            print("Generator weights loaded")
        else:
            raise FileNotFoundError(f"Cannot find generator weights at {generator_weights_path}")

        discriminator_weights_path = os.path.join(self.model_save_dir, f'{load_resolution}x{load_resolution}_generator.h5')
        if os.path.isfile(discriminator_weights_path):
            self.model.Discriminator.load_weights(discriminator_weights_path, by_name=True)
            print("Discriminator weights loaded")
        else:
            raise FileNotFoundError(f"Cannot find discriminator weights at {discriminator_weights_path}")

    @staticmethod
    def calculate_batch_size(resolution):
        '''
        Public static method
        Compute batch size

        :params:
            int resolution: Resolution to compute batch size with
        '''
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
        '''
        Property Alpha. Defines final layer bypass percentage
        '''
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        '''
        Update the value of the merging factor alpha
        
        :params:
            float alpha: merging factor, must be in [0, 1]
        '''
        if value < 0 or value > 1:
            raise ValueError("alpha must be in [0,1]")
        
        if hasattr(self, 'alpha'):
            if self.alpha != value:
                self.model.alpha = value
                self._alpha = value
        else:
            self.model.alpha = value
            self._alpha = value

    def __train_epoch(
            self,
            dataset,
            resolution,
            epoch,
            verbose=False
        ):

        '''
        Train 1 epoch for the PGGAN
        
        :params:
            tf.keras.Dataset dataset: keras dataset containing images in [b, h, w, c] 
            int resolution: Current Resolution for training
            int epoch: Current epoch, for logging purposes
            bool verbose: Verbosity

        return bool: If successful 
        '''

        if verbose:
            print(f'Start of epoch {epoch:d}')
            print(f'Current alpha: {self.alpha:.3f}')
            print(f'Current resolution: {resolution} * {resolution}')

        start = time.time()
        real_image_batch_visualise = None

        for _, (real_image_batch) in enumerate(dataset):
            # Add to overall step and set as checkpoint step variable
            self.overall_steps += 1
            self.checkpoint.step.assign(self.overall_steps)

            # Generate Noise Vector and input into train step
            noise = tf.random.normal((self.resolution_batch_size, self.latent_dim))
            self.discriminator_train_steps[str(resolution)](real_image_batch, noise, verbose=verbose)
            self.generator_train_steps[str(resolution)](noise, verbose=verbose)
            
            # update alpha
            if resolution > 4 and self.hard_start_steps <= 0:
                self.alpha = min(1., self.alpha + self.resolution_alpha_increment)

            # Check that batch size is correct
            if real_image_batch.shape[0] < self.resolution_batch_size:
                raise ValueError('Image batch shape less than resolution batch size')

            # Randomly choose an image batch for visualisation                
            if real_image_batch_visualise is None:
                real_image_batch_visualise = real_image_batch
            elif np.random.uniform() < 0.3:
                real_image_batch_visualise = real_image_batch

        # Write logged losses
        if epoch % self.loss_iter_evaluation == 0:
            alpha_tensor = tf.constant(np.repeat(self.alpha, self.resolution_batch_size).reshape(self.resolution_batch_size, 1), dtype=tf.float32)
            
            # Log alpha
            self.metrics['alpha'](self.alpha)

            # Log generator logs
            with self.gen_summary_writer.as_default():
                tf.summary.scalar('generator_wasserstein_loss', self.metrics['generator_wasserstein_loss'].result(), step=self.overall_steps)
                tf.summary.scalar('generator_loss', self.metrics['generator_loss'].result(), step=self.overall_steps)
                tf.summary.scalar('alpha', self.metrics['alpha'].result(), step=self.overall_steps)

            # Log discriminator logs
            with self.dis_summary_writer.as_default():
                tf.summary.scalar('discriminator_wasserstein_loss_real', self.metrics['discriminator_wasserstein_loss_real'].result(), step=self.overall_steps)
                tf.summary.scalar('discriminator_wasserstein_loss_fake', self.metrics['discriminator_wasserstein_loss_fake'].result(), step=self.overall_steps)
                tf.summary.scalar('discriminator_wasserstein_gradient_penalty', self.metrics['discriminator_wasserstein_gradient_penalty'].result(), step=self.overall_steps)
                tf.summary.scalar('discriminator_loss', self.metrics['discriminator_loss'].result(), step=self.overall_steps)

            # Save Images
            predicted_image = self.model.Generator((noise,alpha_tensor), training=False)
            predicted_image = predicted_image[:, :, :, :]* 0.5 + 0.5
            with self.gen_summary_writer.as_default():
                tf.summary.image('Generated Images', predicted_image, max_outputs=16, step=self.overall_steps)

            # Take a look at real images
            real_image_batch_visualise = real_image_batch_visualise[:, :, :, :]* 0.5 + 0.5
            with self.dis_summary_writer.as_default():
                tf.summary.image('Real Images', real_image_batch_visualise, max_outputs=5, step=self.overall_steps)

            # Generate and save images
            self.__generate_and_save_images(epoch, figure_size=(6,6), subplot=(3,3), save=True)

        # Save Checkpoint
        if epoch % self.save_iter == 0:
            self.__save_check_point(resolution, verbose=True, save_to_gdrive=self.colab, g_drive_path = self.g_drive_path)

        # Reset Losses
        for k in self.metrics:
            self.metrics[k].reset_states()

        # Assign epoch to checkpoint epoch variable
        self.checkpoint.epoch.assign(epoch)

        print(f'Time for epoch {epoch} is {time.time()-start:.3f} sec. Training time: {time.time()-self.train_start_time:.3f}. Alpha = {self.alpha:.5f}')

        # Configure alpha for hard starts
        if self.hard_start:
            if self.hard_start_steps > 0:
                self.hard_start_steps -= 1
                print('Hard start steps left: ', self.hard_start_steps)
            if self.hard_start_steps == 0:
                self.hard_start_steps -= 1
                self.alpha = min(1., (self.start_epoch - 1) % self.epochs * len(dataset) * self.resolution_alpha_increment)

        if verbose:
            print('Completed')

        return True

    
    def train(
            self,
            restore=False,
            colab=False,
            load_from_g_drive=False,
            verbose=False,
            g_drive_path = '/content/drive/My Drive/CML'
        ):
        '''
        Main method for training. Trains PGGAN from start till the final resolution

        :params:
            bool restore: If restoring from checkpoint and weights
            bool colab: If training in colab environment
            bool load_from_g_drive: If loading checkpoint and weights from GDrive storage. Only applicable in Colab
            bool verbose: Verbosity of training
            str g_drive_path: Path to folder where checkpoints.zip and weights.zip can be found
        
        :return bool: True if successful
        '''

        self.colab = colab
        self.train_start_time = time.time()
        self.g_drive_path = g_drive_path
        self.loaded = False

        # Restore weights and checkpoints if necessary
        if restore:
            self.__load_saved_training(load_from_g_drive=load_from_g_drive)
            self.loaded = True

        # Start off on the start resolution
        resolution = self.start_resolution
        while resolution <= self.stop_resolution:
            print(f'Size {resolution}x{resolution} training begins')

            # Define specific paths
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

            # Create training steps
            self.discriminator_train_steps[str(resolution)] = copy(self.discriminator_train_step)
            self.generator_train_steps[str(resolution)] = copy(self.generator_train_step)

            training_steps = len(train_dataset)
            # Fade in half of switch_res_every_n_epoch epoch, and stablize another half
            self.resolution_alpha_increment = 1. / (self.epochs / 5 * training_steps)

            # If hard start, set alpha to 1 for hard start steps
            if self.hard_start and not self.loaded:
                self.hard_start_steps = 10
                self.alpha = 0.9
            else:
                self.hard_start_steps = -1
                self.alpha = min(1., (self.start_epoch - 1) % self.epochs * training_steps * self.resolution_alpha_increment)
            
            # Make sure starting epoch is more than epochs to train for
            assert self.start_epoch <= self.epochs, f'Start epochs {self.start_epoch} should be less than epochs: {self.epochs}'
            
            # Train for self.epochs epoch
            for epoch in range(self.start_epoch, self.epochs + 1):
                self.__train_epoch(train_dataset, resolution, epoch, verbose=verbose)

            self.__save_check_point(resolution, verbose=True, save_to_gdrive=self.colab, g_drive_path = self.g_drive_path)
            
            # Double model resolution
            if resolution != self.stop_resolution:
                self.model.double_resolution()
                self.load_weights(resolution)
                self.start_epoch = 1
            resolution *= 2
            self.loaded = False

        return True

    @tf.function
    def discriminator_train_step(
            self,
            real_images,
            noise,
            verbose=False
        ):
        '''
        Discriminator train step. Trains discriminator for one step. This is a tf.function, so need to create one for every resolution

        :params:
            tf.Tensor real_images: real_images in shape [b, h, w, c]
            tf.Tensor noise: Noise tensor
            bool verbose: Verbosity
        '''
        # Generate epsilon for reduction of Lipschitz function gradient
        epsilon = tf.random.uniform(shape=[self.resolution_batch_size, 1, 1, 1], minval=0, maxval=1)

        # Generate alpha tensor
        alpha_tensor = tf.constant(np.repeat(self.alpha, self.resolution_batch_size).reshape(self.resolution_batch_size, 1), dtype=tf.float32)

        # Record discriminator gradient
        with tf.GradientTape(persistent=True) as d_tape:

            # Record gradient of discriminator Wasserstein output
            with tf.GradientTape() as gp_tape:
                generated_images = self.model.Generator((noise, alpha_tensor), training=True)
                generated_images_mixed = epsilon * tf.dtypes.cast(real_images, tf.float32) + ((1 - epsilon) * generated_images)
                fake_mixed_pred = self.model.Discriminator((generated_images_mixed,alpha_tensor), training=True)
                
            # Compute gradient penalty
            grads = gp_tape.gradient(fake_mixed_pred, generated_images_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            discriminator_gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            if verbose:
                print('Obtained Wasserstein Gradient Penalty for Discriminator')
            
            # Compute Wasserstein loss for discriminator
            discriminator_wloss_real = -tf.math.reduce_mean(self.model.Discriminator((real_images,alpha_tensor), training=True))
            if verbose:
                print('Obtained Wasserstein Loss for Discriminator on REAL images')
            discriminator_wloss_fake = tf.math.reduce_mean(self.model.Discriminator((generated_images,alpha_tensor), training=True))
            if verbose:
                print('Obtained Wasserstein Loss for Discriminator on FAKE images')

            total_discriminator_loss = discriminator_wloss_real + discriminator_wloss_fake + self.config.lambdaGP*discriminator_gradient_penalty

            # Log losses
            self.metrics['discriminator_wasserstein_loss_real'](discriminator_wloss_real)
            self.metrics['discriminator_wasserstein_loss_fake'](discriminator_wloss_fake)
            self.metrics['discriminator_wasserstein_gradient_penalty'](discriminator_gradient_penalty)
            self.metrics['discriminator_loss'](total_discriminator_loss)

        # Calculate the gradients for discriminator using backpropagation
        gradients_of_discriminator = d_tape.gradient(total_discriminator_loss, self.model.Discriminator.trainable_variables)
        if verbose:
            print('Computed discriminator loss gradients')

        # Apply gradient descent using optimizer
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.Discriminator.trainable_variables))
        if verbose:
            print('Applied discriminator loss gradients')

    @tf.function
    def generator_train_step(self, noise, verbose=False, return_generated_images=False):
        '''
        Generator train step. Trains generator for one step. This is a tf.function, so need to create one for every resolution

        :params:
            tf.Tensor noise: Noise tensor
            bool verbose: Verbosity
            bool return_generated_images: Whether to return generated images
        
        :return tf.Tensor: generated images in shape [b, h, w, c]
        '''

        # Define alpha tensor
        alpha_tensor = tf.constant(np.repeat(self.alpha, self.resolution_batch_size).reshape(self.resolution_batch_size, 1), dtype=tf.float32)

        # Record generator gradients
        with tf.GradientTape() as g_tape:
            generated_images = self.model.Generator((noise,alpha_tensor), training=True)
            fake_predictions = self.model.Discriminator((generated_images,alpha_tensor), training=True)

            # Compute generator Wasserstein Loss
            generator_wloss_fake = -tf.math.reduce_mean(fake_predictions)
            if verbose:
                print('Obtained Wasserstein Loss for Generator on FAKE images')

            total_generator_loss = generator_wloss_fake

            # Log losses
            self.metrics['generator_wasserstein_loss'](generator_wloss_fake)
            self.metrics['generator_loss'](total_generator_loss)
        
        # Calculate the gradients for discriminator using back propagation
        gradients_of_generator = g_tape.gradient(total_generator_loss, self.model.Generator.trainable_variables)
        if verbose:
            print('Computed generator loss gradients')
        
        # Apply gradient descent using the optimizer
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.Generator.trainable_variables))
        if verbose:
            print('Applied generator loss gradients')

        if return_generated_images:
            return generated_images