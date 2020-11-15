import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import datetime as dt
import argparse
import numpy as np

from PIL import Image
from utils.preprocessing import random_image_sample, get_image_dataset, get_cgan_image_datasets
from utils.models import Generator, Discriminator, CGGenerator, CGDiscriminator, get_classifier
from utils.train import train, CycleGANTrainer, ClassifierTrainer

import matplotlib.pyplot as plt

from IPython import display

def get_options():
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size =')
    parser.add_argument('--img_height', default=32, type=int, help='Image Height and Width')
    parser.add_argument('--latent_dim', default=100, type=int, help='Dimension of latent dimension')
    parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to train on')
    parser.add_argument('--save_step', default=100, type=int, help='Number of epochs before saving')
    parser.add_argument('--saveimg_step', default=10, type=int, help='Number of epochs before saving an image')

    # Optimizer options
    parser.add_argument('--glr', default=1e-4, type=float, help='Learning rate for generator')
    parser.add_argument('--dlr', default=1e-4, type=float, help='Learning rate for discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam optimizer beta1.')
    parser.add_argument('--beta2', default=0.5, type=float, help='Adam optimizer beta2.')
    parser.add_argument('--gamma', default=0.9999, type=float, help='Exponential LR scheduler gamma discount factor.')

    # CGAN options
    parser.add_argument('--cgan', action='store_true', help='Run Cycle GAN')

    # Classifier Options
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--classifier', action='store_true', help='Train Classifier')
    parser.add_argument('--infer', action='store_true', help='Infer Data From Trained Classifier')
    
    # Restore options
    parser.add_argument('--cgan_restore', action='store_true', help='Restore Cycle GAN from checkpoint')
    parser.add_argument('--restore_gdrive', action='store_true', help='Restore from last checkpoint in gdrive')
    parser.add_argument('--clean_data_dir', action='store_true', help='Remove all images in data dir with less than 128 pixel H/W')

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
    assert opt.gamma > 0

    return opt

if __name__ == '__main__':
    opt = get_options()

    try:
        from google.colab import drive
        colab = True
        print('Training in colab environement')
    except ModuleNotFoundError:
        colab = False

    if opt.cgan:
        data_directory = os.path.join(os.getcwd(),'data','FACADES_UNPAIRED')

        # Rename files
        directories = [
            os.path.join(data_directory,'unpaired_train_A'),
            os.path.join(data_directory,'unpaired_train_B'),
            os.path.join(data_directory,'unpaired_test_A'),
            os.path.join(data_directory,'unpaired_test_B')
        ]

        file_paths = []
        for directory in directories:
            for (dirpath, dirnames, filenames) in os.walk(directory):
                file_paths.extend(filenames)
                break
            for file_path in file_paths:
                file_path = os.path.join(directory, file_path)
                if '.jpg' in os.path.splitext(file_path)[1]:
                    base = os.path.splitext(file_path)[0]
                    os.rename(file_path, base + '.jpeg')


        data_patterns = [
            os.path.join(data_directory,'unpaired_train_A','*.jpeg'),
            os.path.join(data_directory,'unpaired_train_B','*.jpeg'),
            os.path.join(data_directory,'unpaired_test_A','*.jpeg'),
            os.path.join(data_directory,'unpaired_test_B','*.jpeg')
        ]

        IMAGE_HEIGHT=128

        if opt.clean_data_dir:
            for data_dir in data_patterns:
                pic_list = glob.glob(data_dir)
                pic_image_length = len(pic_list)
                count = 0
                for fp in pic_list:
                    shape = np.array(Image.open(fp)).shape
                    if shape[0] < IMAGE_HEIGHT or shape[1] < IMAGE_HEIGHT:
                        count+=1
                        os.remove(fp)
                print(f'Removed {count} images. Left {pic_image_length-count}')

        train_datasetA =  get_cgan_image_datasets(os.path.join(data_directory,'unpaired_train_A','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=True)
        train_datasetB = get_cgan_image_datasets(os.path.join(data_directory,'unpaired_train_B','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=False)

        test_datasetA =  get_cgan_image_datasets(os.path.join(data_directory,'unpaired_test_A','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=True)
        test_datasetB = get_cgan_image_datasets(os.path.join(data_directory,'unpaired_test_B','*.jpeg'), IMAGE_HEIGHT, IMAGE_HEIGHT, 1, train=False)

        generator_a2b = CGGenerator()
        generator_b2a = CGGenerator()
        discriminator_a = CGDiscriminator()
        discriminator_b = CGDiscriminator()

        generator_a2b.build((1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3))
        generator_b2a.build((1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3))
        discriminator_a.build((1, IMAGE_HEIGHT,IMAGE_HEIGHT,3))
        discriminator_b.build((1, IMAGE_HEIGHT,IMAGE_HEIGHT,3))

        generator_a2b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        generator_b2a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        cgan_trainer = CycleGANTrainer(
            train_datasets = (train_datasetA, train_datasetB),
            test_datasets = (test_datasetA, test_datasetB),
            generators = (generator_a2b, generator_b2a),
            discriminators = (discriminator_a, discriminator_b),
            discriminator_optimizers = (discriminator_a_optimizer, discriminator_b_optimizer),
            generator_optimizers = (generator_a2b_optimizer, generator_b2a_optimizer),
            epochs=opt.epochs,
        )

        cgan_trainer.train(restore = opt.cgan_restore, colab = colab, load_from_g_drive=opt.restore_gdrive, save_to_gdrive=True, g_drive_path = '/content/drive/My Drive/CML')
    elif opt.classifier:

        print('Training Classifier')
        data_directory = os.path.join(os.getcwd(),'classifier_data')

        train_ds = keras.preprocessing.image_dataset_from_directory(
            data_directory,
            labels='inferred',
            color_mode='rgb',
            seed=1234,
            validation_split=0.2,
            subset="training",
            image_size=(opt.img_height,opt.img_height),
            batch_size=opt.batch_size,
        )

        val_ds = keras.preprocessing.image_dataset_from_directory(
            data_directory,
            labels='inferred',
            color_mode='rgb',
            seed=1234,
            validation_split=0.2,
            subset="validation",
            image_size=(opt.img_height,opt.img_height),
            batch_size=opt.batch_size,
        )

        folders = 0

        for _, dirnames, _ in os.walk(data_directory):
            folders += len(dirnames)

        classifier_net = get_classifier((opt.img_height, opt.img_height, 3), num_classes=folders)

        def lr_schedule(epoch):
            """
            Returns a custom learning rate that decreases as epochs progress.
            """
            learning_rate = opt.lr
            if epoch > 25:
                learning_rate = opt.lr/10
            if epoch > 50:
                learning_rate = opt.lr/100
            if epoch > 75:
                learning_rate = opt.lr/1000

            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            return learning_rate

        trainer = ClassifierTrainer(train_ds, val_ds, classifier_net, tf.keras.optimizers.Adam(learning_rate=opt.lr), lr_schedule)

        if opt.infer:
            infer_dir = os.path.join(os.getcwd(),'classifier_infer_data')
            class_names = os.listdir(data_directory)
            trainer.infer(infer_dir, opt.img_height, opt.img_height, class_names)
        else:
            trainer.train(opt.epochs, opt.batch_size)

    else:
        data_directory = os.path.join(os.getcwd(),'data')

        train_dataset = get_image_dataset(os.path.join(data_directory,'google_pavilion','*.jpeg'), opt.img_height, opt.img_height, opt.batch_size)

        if opt.img_height == 256:
            print('Using upscaled DCGAN')
            generator = Generator(latent_dim = 512, upscale=True)
            discriminator = Discriminator(upscale=True)
        else:
            generator = Generator(latent_dim = opt.latent_dim)
            discriminator = Discriminator()

        generator.build((opt.batch_size,opt.latent_dim))
        discriminator.build((opt.batch_size,opt.img_height,opt.img_height,3))

        generator_optimizer = keras.optimizers.Adam(opt.glr)
        discriminator_optimizer = keras.optimizers.Adam(opt.dlr)

        train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, opt.epochs, opt.batch_size, opt.latent_dim, data_directory,False,opt.save_step,opt.saveimg_step)





# data_dir = os.getcwd()

# dataset_dir = data_dir+"/inputs/FACADES_A"
# output_dir = data_dir+"/outputs/GAN"
# log_dir = data_dir+"/logs/GAN"

# buffer_size = 506 # based on the size of your dataset (400 + 106 = 506)
# batch_size = 32 # keep to powers of 2 #32

# img_width = 256
# img_height = 256

# img_w = 32 #downsample upon data loading
# img_h = 32

# aug_w = 36 #upsample for data augmentation
# aug_h = 36

# latent_dim = 100
# epochs = 50 #5K, 10K or 20K

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     dataset_dir, labels='inferred',label_mode=None, color_mode='rgb',batch_size=batch_size, image_size=(img_h, img_w),shuffle=True,interpolation='bilinear')

# plt.figure(figsize=(10,10))
# for images in train_ds.take(1):
#     for i in range(16):
#         ax = plt.subplot(4, 4, i + 1)
#         plt.imshow(images[i].numpy().squeeze()/255.0)
#         plt.axis("off")

# # Use this if you want to include data-augmentation
# # def normalize(img):
# #   # resize for cropping
# #   img = tf.image.resize(img,[aug_h,aug_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# #   # random crop
# #   img = tf.image.random_crop(img, size=[1,img_h,img_w,3])
# #   # random flip between left/right
# #   if tf.random.uniform(())>0.5:
# #     img = tf.image.flip_left_right(img)
# #   # normalise to range between -1 & 1
# #   img = tf.cast(img,tf.float32)
# #   img = (img/127.5)-1
# #   return img

# # Use this if you want to exclude data-augmentation
# def normalize(img):
#   # normalise to range between -1 & 1
#   img = tf.cast(img,tf.float32)
#   img = (img/127.5)-1
#   return img

# train_ds = train_ds.map(normalize)

# for i in train_ds.take(1):
#   print(i.shape)

# #Generator

# def build_generator():
#   model = keras.Sequential()
#   model.add(layers.Input(shape=(latent_dim,)))
#   model.add(layers.Dense(8*8*256))
#   model.add(layers.BatchNormalization())
#   model.add(layers.LeakyReLU())

#   model.add(layers.Reshape((8,8,256)))

#   model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))
#   model.add(layers.BatchNormalization())
#   model.add(layers.LeakyReLU())

#   model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
#   model.add(layers.BatchNormalization())
#   model.add(layers.LeakyReLU())

#   model.add(layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))

#   for layer in model.layers:
#     print(layer.output_shape)

#   return model

# generator = build_generator()
# generator.summary()

# # To show a generated image from an UNTRAINED generator
# noise = tf.random.normal([1,100])
# gen_img = generator(noise,training=False)
# gen_img_ = gen_img.numpy().squeeze()
# print(gen_img.shape)
# print(gen_img_.shape)
# ax = plt.figure()
# plt.imshow(gen_img_ * 0.5 + 0.5) # map from range(-1,1) to range(0,1)

# #Discriminator

# def build_discriminator():
#   model = keras.Sequential()
#   model.add(layers.Input(shape=(32,32,3)))
#   model.add(layers.Conv2D(32,(5,5),strides=(2,2),padding='same'))
#   model.add(layers.LeakyReLU())
#   model.add(layers.Dropout(0.3))

#   model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same'))
#   model.add(layers.LeakyReLU())
#   model.add(layers.Dropout(0.3))

#   model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
#   model.add(layers.LeakyReLU())
#   model.add(layers.Dropout(0.3))

#   model.add(layers.Flatten())
#   model.add(layers.Dense(1))

#   for layer in model.layers:
#     print(layer.output_shape)

#   return model

# discriminator = build_discriminator()
# discriminator.summary()

# # To show a true/fake prediction from an untrained discriminator. True(+), False(-)
# decision = discriminator(gen_img)
# print(decision)

# #Discriminator loss

# cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
# def discriminator_loss(real_output, fake_output):
#   real_loss = cross_entropy(tf.random.uniform([real_output.shape[0]],0.7,1.2), real_output) # set noise to 1
#   fake_loss = cross_entropy(tf.random.uniform([fake_output.shape[0]],0,0.3), fake_output) # set noise to 0
#   total_loss = real_loss + fake_loss
#   return total_loss

# #Generator loss

# def generator_loss(fake_output):
#   return cross_entropy(tf.random.uniform([fake_output.shape[0]],0.7,1.2), fake_output) # set noise to 1


# #Optimiser

# generator_optimizer = keras.optimizers.Adam(1e-4) # 0.0001
# discriminator_optimizer = keras.optimizers.Adam(1e-4)

# #Metrics

# # define metrics
# sgen_loss = tf.keras.metrics.Mean('sgen_loss', dtype=tf.float32)
# sdis_loss = tf.keras.metrics.Mean('sdis_loss', dtype=tf.float32)
# sdis_acc = tf.keras.metrics.BinaryAccuracy('sdis_acc')

# #Save Checkpoints

# checkpoint_path = data_dir+"/training/GAN/ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# checkpoint = tf.train.Checkpoint(step=tf.Variable(1),generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# #Training (with gradient tape)
# num_gen_samples = 16
# seed = tf.random.normal([num_gen_samples,latent_dim])

# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([batch_size, latent_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#       generated_images = generator(noise, training=True)

#       real_output = discriminator(images, training=True)
#       fake_output = discriminator(generated_images, training=True)

#       gen_loss = generator_loss(fake_output)
#       disc_loss = discriminator_loss(real_output, fake_output)

#       sdis_loss(disc_loss)
#       sgen_loss(gen_loss)
#       sdis_acc(tf.ones_like(real_output), real_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# #Uncomment if new file
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# #Uncomment and use the same name of your last folder if continue training (i.e. google disconnected)
# # current_time = "20201005-155429"

# gen_log_dir = log_dir+'/gradient_tape/' + current_time + '/gen'
# dis_log_dir = log_dir+'/gradient_tape/' + current_time + '/dis'

# gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
# dis_summary_writer = tf.summary.create_file_writer(dis_log_dir)

# def train(dataset, epochs, restore=False, save_step=100,saveimg_step=10):

#   if restore:
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#     print("Restored from epoch{}".format(int(checkpoint.step)))
#     add_step=int(checkpoint.step)
#     print("Restore")
#   else:
#     add_step=0
#     print("Fresh")

#   for epoch in range(epochs):

#     if restore:
#       step=int(checkpoint.step)+epoch
#     else:
#       step=epoch

#     start = time.time()

#     for image_batch in dataset:
#       train_step(image_batch)

#     with gen_summary_writer.as_default():
#       tf.summary.scalar('sgen_loss', sgen_loss.result(), step=step)

#     with dis_summary_writer.as_default():
#       tf.summary.scalar('sdis_loss', sdis_loss.result(), step=step)
#       tf.summary.scalar('sdis_acc', sdis_acc.result(), step=step)

#     display.clear_output(wait=True)
#     generate_and_save_images(generator,
#                               epoch + 1 + add_step,
#                               seed)

#     if (epoch + 1) % save_step == 0:
#       checkpoint.step.assign_add(save_step)
#       checkpoint.save(file_prefix = checkpoint_path)
#       print (int(checkpoint.step))
#     template = 'Epoch {}, Generator Loss: {}, Discriminator Loss: {}, Discriminator Accuracy: {}'
#     print (template.format(epoch+1,
#                             sgen_loss.result(),
#                             sdis_loss.result(),
#                             sdis_acc.result()))
#     print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#     sgen_loss.reset_states()
#     sgen_loss.reset_states()
#     sdis_loss.reset_states()

#   # Generate after the final epoch
#   display.clear_output(wait=True)
#   generate_and_save_images(generator,epochs,seed,saveimg_step)

#   def generate_and_save_images(model, epoch, test_input,saveimg_step=1):
#   # Notice `training` is set to False.
#   # This is so all layers run in inference mode (batchnorm).
#     predictions = model(test_input, training=False)

#     if epoch%saveimg_step==0:

#       fig = plt.figure(figsize=(10,10))

#       for i in range(predictions.shape[0]):
#           plt.subplot(4, 4, i+1)
#           plt.imshow(predictions[i, :, :, :]* 0.5 + 0.5) # map from range(-1,1) to range(0,1)

#           plt.axis('off')

#     plt.savefig(output_dir+'/image_at_epoch_{:04d}.png'.format(epoch))
#     plt.show()

#   %tensorboard --logdir "/content/drive/My Drive/Colab Notebooks/CML_2020_Codes/week_04/logs/GAN" # full path

#   # set restore=False first time training
# train(train_ds, epochs, restore=False, save_step=10,saveimg_step=1)

# # set restore=True if continue from last training checkpoint
# # train(train_ds, epochs, restore=True, save_step=10,saveimg_step=1) # to avoid filling up your G-Drivw change save_step=100 and saveimg_step=100

# # Display a single image using the epoch number
# def display_image(epoch_no):
#   return PIL.Image.open(output_dir+'/image_at_epoch_{:04d}.png'.format(epoch_no))

# display_image(epochs)

# Use this to simply restore to the latest checkpoint (if you don't want to continue from your last training session).
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# # To show a generated image from a TRAINED generator
# noise = tf.random.normal([1,100])
# gen_img = generator(noise,training=False)
# gen_img_ = gen_img.numpy().squeeze()
# print(gen_img.shape)
# print(gen_img_.shape)
# ax = plt.figure()
# plt.imshow(gen_img_ * 0.5 + 0.5) # map from range(-1,1) to range(0,1)
