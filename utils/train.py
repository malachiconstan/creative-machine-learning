import time
import os
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from IPython import display

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.random.uniform([real_output.shape[0]],0.7,1.2), real_output) # set noise to 1
    fake_loss = cross_entropy(tf.random.uniform([fake_output.shape[0]],0,0.3), fake_output) # set noise to 0
    total_loss = real_loss + fake_loss
    return total_loss
#
# def discriminator_loss(real_output, fake_output):
#     real_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([real_output.shape[0],1],0.7,1.2), real_output, from_logits=True) # set noise to 1
#     fake_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0,0.3), fake_output, from_logits=True) # set noise to 0
#     total_loss = real_loss + fake_loss
#     return total_loss

# def generator_loss(fake_output):
#     return tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0.7,1.2), fake_output, from_logits=True) # set noise to 1

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.random.uniform([fake_output.shape[0]],0.7,1.2), fake_output) # set noise to 1

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
    generate_and_save_images(generator,epoch,seed,output_dir)
