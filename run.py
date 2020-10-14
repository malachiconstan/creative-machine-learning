import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import datetime as dt

from PIL import Image
from utils.preprocessing import random_image_sample, get_image_dataset
from utils.models import Generator, Discriminator
from utils.train import train

import matplotlib.pyplot as plt

from IPython import display

if __name__ == '__main__':
  BATCH_SIZE = 32
  IMG_HEIGHT = 32
  IMG_WIDTH = 32
  LATENT_DIM = 100
  EPOCHS = 1

  data_directory = os.path.join(os.getcwd(),'data')
  image_path_pattern = os.path.join(data_directory,'gallery_pavilion','*.jpeg')

  train_dataset = get_image_dataset(os.path.join(data_directory,'gallery_pavilion','*.jpeg'), IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

  generator = Generator(latent_dim = LATENT_DIM)
  discriminator = Discriminator()

  generator.build((100,LATENT_DIM))
  discriminator.build((10,32,32,3))

  generator_optimizer = keras.optimizers.Adam(1e-4)
  discriminator_optimizer = keras.optimizers.Adam(1e-4)

  train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, EPOCHS, BATCH_SIZE, LATENT_DIM, data_directory, False, 10, 1)





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
