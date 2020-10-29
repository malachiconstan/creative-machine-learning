import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

def random_image_sample(paths, chosen_images = 10):
    '''
    Displays random images from the image path
    :params
        List paths: List of image paths
        int chosen_images: number of images to display
    :return
        Void
    '''
    chosen_image_paths = np.random.choice(np.array(paths),chosen_images,replace=False)

    fig, ax = plt.subplots(1, chosen_images, figsize=(20, 2))
    for i in range(1):
        for j in range(chosen_images):
            ax[j].imshow(Image.open(chosen_image_paths[j]))
            ax[j].xaxis.set_visible(False)
            ax[j].yaxis.set_visible(False)
    fig.suptitle('Sampled Images')

def decode_img(img,img_height,img_width, augment):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    if augment:
        # img = tf.image.resize(img,[int(1.5*img_height),int(1.5*img_width)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if tf.random.uniform(())>0.5:
            img = tf.image.flip_left_right(img)
    else:    
        img = tf.image.central_crop(img, 1)
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path,img_height,img_width,normalize=True,augment=True):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img,img_height,img_width,augment=augment)
    # resize for cropping
    
    # random crop
    
    # random flip between left/right
    
    if normalize:
        img = tf.cast(img,tf.float32)
        img = (img/127.5)-1
    return img


def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_image_dataset(file_pattern,img_height=180,img_width=180,batch_size=32,normalize=True,augment=True):
    '''
    Function to return a train dataset from glob file pattern
    '''

    list_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    image_count = len(list_dataset)
    list_dataset = list_dataset.shuffle(image_count, reshuffle_each_iteration=False)
    train_dataset = list_dataset.skip(0) # No validation required for GAN
    train_dataset = train_dataset.map(lambda x: process_path(x,img_height,img_width,normalize,augment), num_parallel_calls=AUTOTUNE)
    train_dataset = configure_for_performance(train_dataset, batch_size)

    return train_dataset

# Functions for cycle GAN
def load_single(img_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    return img

def random_jitter(img,img_height,img_width):
    jit_img = tf.image.resize(img,[int(1.25*img_height),int(1.25*img_width)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    jit_img = tf.image.random_crop(img, size=[img_height,img_width,3])
    jit_img = tf.image.random_flip_left_right(jit_img)
    return jit_img

def load_img(img_file,img_height,img_width,train,normalize):
    img = load_single(img_file)
    if train:
        # img = tf.image.random_flip_left_right(img)
        img = random_jitter(img,img_height,img_width)

    if normalize:
        img = (img/127.5)-1
    return img

def get_cgan_image_datasets(file_pattern,img_height=180,img_width=180,batch_size=32,normalize=True,train=True):
    list_dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = list_dataset.map(lambda x: load_img(x,img_height,img_width,train=train,normalize=normalize), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(batch_size).batch(batch_size)
    return dataset
