import tensorflow as tf
import tensorflow.keras as keras

# import tensorflow.compat.v1 as tfv1
# tf.disable_v2_behavior()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime as dt
import json

from utils.pgan.losses import wgan_loss, discriminator_loss, generator_loss
from utils.pgan.net import PGGenerator, PGDiscriminator
from utils.pgan.utils import ImageLoader

from utils.preprocessing import get_image_dataset

class PGGANTrainer(object):
    def __init__(self,
                cfg,
                discriminator_optimizer,
                generator_optimizer,
                main_directory=os.getcwd(),
                # loss_iter_evaluation=200,
                # save_iter=5000,
                model_label='PGGAN'):
        # super(PGGAN, self).__init__(name = model_label, **kwargs)

        # Define directories
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = os.path.join(main_directory,'pggan_logs')
        self.image_save_dir = os.path.join(main_directory,'pggan_imgs')
        self.gen_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'gen')
        self.dis_log_dir = os.path.join(self.log_dir,'gradient_tape',current_time,'dis')
        self.checkpoint_dir = os.path.join(main_directory,'pggan_checkpoints')
        self.train_config_path = os.path.join(self.checkpoint_dir, f'{model_label}_' + "_train_config.json")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.cfg = cfg
        # self.tf_placeholders = {}
        # self.create_tf_placeholders()
        self.d_train_op, self.g_train_op = None, None
        self.ema_op, self.ema_vars = None, {}
        # self.d_loss, self.g_loss = None, None
        self.gen_images, self.eval_op = None, None
        self.image_loader = ImageLoader(self.cfg)
        self.train_dataset = get_image_dataset(os.path.join(os.getcwd(),'data','google_pavilion','*.jpeg'),img_height=self.cfg.resolution,img_width=self.cfg.resolution,batch_size=self.cfg.batch_size,normalize=True,augment=True) #self.image_loader.get_image_dataset()

        # Initialise Models
        self.Generator = PGGenerator(cfg)
        self.Discriminator = PGDiscriminator(cfg)

        # Define Parameters
        self.start_scale = 0
        self.start_iter = 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.h, self.w, self.c = self.cfg.input_shape
        self.z_dim = self.cfg.z_dim
        self.n_critic = self.cfg.n_critic
        self.n_iters = self.cfg.n_iters
        self.batch_size = self.cfg.batch_size
        self.transition = self.cfg.transition
        self.alpha = self.cfg.fade_alpha

        # Define Optimizers
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)

        # Logging
        self.display_period = self.cfg.display_period
        self.metrics = dict(
            # discriminator_wasserstein_loss_real = tf.keras.metrics.Mean('discriminator_wasserstein_loss_real', dtype=tf.float32),
            discriminator_wasserstein_loss = tf.keras.metrics.Mean('discriminator_wasserstein_loss', dtype=tf.float32),
            discriminator_wasserstein_gradient_penalty = tf.keras.metrics.Mean('discriminator_wasserstein_gradient_penalty', dtype=tf.float32),
            # generator_wasserstein_loss = tf.keras.metrics.Mean('generator_wasserstein_loss', dtype=tf.float32),
            discriminator_epsilon_loss = tf.keras.metrics.Mean('discriminator_epsilon_loss', dtype=tf.float32),
            discriminator_loss = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32),
            generator_loss = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32),
        )

        self.gen_summary_writer = tf.summary.create_file_writer(self.gen_log_dir)
        self.dis_summary_writer = tf.summary.create_file_writer(self.dis_log_dir)

        # Low-res layers
        self.avgpool2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
        self.upsampling2d = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')     

    def resize_image(self, image, verbose=True):
        input_size = image.shape[1]
        res = self.cfg.resolution

        if verbose:
            print('Input size is :', input_size)
            print('Resolution is :', res)

        if input_size == res:
            return image
        new_size = [res, res]
        new_img = tf.image.resize(image, size=new_size, method='nearest')
        # new_img = tf.image.resize_nearest_neighbor(image, size=new_size)
        if self.transition:
            # alpha = self.tf_placeholders['alpha']
            low_res_img = tf.nn.avg_pool2d(new_img, ksize=2, strides=2, padding='valid')
            # low_res_img = tf.layers.average_pooling2d(new_img, 2, 2)
            low_res_img = tf.image.resize(low_res_img, size=new_size, method='nearest')
            # low_res_img = tf.image.resize_nearest_neighbor(low_res_img, size=new_size)
            new_img = self.alpha * new_img + (1. - self.alpha) * low_res_img
        return new_img

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


        self._alpha = value
        self.Generator.alpha = value
        self.Discriminator.alpha = value

    def train_step(self, real_images, noise, verbose=False):
        # tf.summary.image('images_real_original_size', images_real, 8)
        # tf.summary.image('images_real', images_real, 8)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # 1. Wasserstein Loss for both generator and discriminator
            real_predictions = self.Discriminator(real_images, training=True)
            generated_images = self.Generator(noise, training=True)
            fake_predictions = self.Discriminator(generated_images, training=True)
            # discriminator_wloss = discriminator_loss(real_predictions, fake_predictions)

            discriminator_wloss, generator_wloss = wgan_loss(real_predictions, fake_predictions)
            # generator_wloss = generator_loss(fake_predictions)

            if verbose:
                print('Obtained Wasserstein Loss for Discriminator and Generator')
            
            # 2. Gradient Penalty Loss
            alpha = tf.random.uniform(shape=[], minval=0., maxval=1.)
            differences = generated_images - real_images
            interpolated = real_images + alpha * differences

            with tf.GradientTape() as gradient_penalty_tape:
                gradient_penalty_tape.watch(interpolated)
                interpolated_prediction = self.Discriminator(interpolated, training=True)

            gradient = gradient_penalty_tape.gradient(interpolated_prediction, [interpolated])[0]
            gradient_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(gradient), axis=[1, 2, 3]))
            gradient_penalty = self.cfg.lambda_gp*tf.math.reduce_mean((gradient_norm/self.cfg.gamma_gp - 1.0) ** 2)

            if verbose:
                print('Obtained WPGAN-GP loss for Discriminator')
            # d_loss, g_loss = None, None
            # if self.cfg.loss_mode == 'js':
            #     smooth_factor = 0.9 if self.cfg.smooth_label else 1.
            #     d_loss, g_loss = losses.js_loss(d_real, d_fake, smooth_factor)
            # elif self.cfg.loss_mode == 'wgan_gp':
            
            # Gradient penalty
            # lambda_gp = self.cfg.lambda_gp
            # gamma_gp = self.cfg.gamma_gp
            # batch_size = self.cfg.batch_size
            # nc = self.cfg.input_shape[-1]
            # res = self.cfg.resolution
            # input_shape = [batch_size, res, res, nc]
            # alpha = tf.random_uniform(shape=input_shape, minval=0., maxval=1.)
            # differences = images_fake - images_real
            # interpolates = images_real + alpha * differences
            # gradients = tf.gradients(
            #     self.build_discriminator(interpolates, reuse=True, training=True),
            #     [interpolates, ])[0]
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            # gradient_penalty = \
            #     lambda_gp * tf.reduce_mean((slopes / gamma_gp - 1.) ** 2)
            # d_loss += gradient_penalty

            if self.cfg.drift_loss:
                drift_loss = self.cfg.eps_drift * tf.math.reduce_mean(tf.nn.l2_loss(real_predictions))
                if verbose:
                    print('Obtained Epsilon loss for Discriminator')

            total_generator_loss = generator_wloss
            total_discriminator_loss = discriminator_wloss + gradient_penalty + drift_loss
            print(f'Discriminator loss: {total_discriminator_loss} Generator loss: {total_generator_loss}')

            # Log losses
            self.metrics['discriminator_wasserstein_loss'](discriminator_wloss)
            # self.metrics['discriminator_wasserstein_gradient_penalty'](0)
            # self.metrics['discriminator_epsilon_loss'](drift_loss)
            self.metrics['discriminator_loss'](total_discriminator_loss)

            self.metrics['generator_loss'](total_generator_loss)
        
        gradients_of_generator = gen_tape.gradient(total_generator_loss, self.Generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(total_discriminator_loss, self.Discriminator.trainable_variables)

        if verbose:
            print('Computed loss gradients')
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        # g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # beta1 = self.cfg.beta1
        # beta2 = self.cfg.beta2
        # learning_rate = self.tf_placeholders['learning_rate']
        # d_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
        # g_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
        # ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # self.ema_op = ema.apply(g_vars)
        # self.ema_vars = {ema.average_name(v): v for v in g_vars}

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.d_train_op = d_solver.minimize(d_loss, var_list=d_vars,
        #                                         global_step=self.global_step)
        #     self.g_train_op = g_solver.minimize(g_loss, var_list=g_vars)
        #     self.d_loss, self.g_loss = d_loss, g_loss

    def save_check_point(self, resolution, global_step, verbose=True, save_to_gdrive=True, g_drive_path = '/content/drive/My Drive/CML'):
        save_path = self.checkpoint_manager.save()
        if verbose:
            print(f"Saved checkpoint for step {int(self.checkpoint.step)}: {save_path}")

        # Tmp Configuration
        # outConfig = {'scale': scale, 'iter': iter}

        # with open(self.temp_config_path, 'w') as fp:
        #     json.dump(outConfig, fp, indent=4)
        
        # if verbose:
        #     print('Saved temp outconfig to: ',self.temp_config_path)

        if save_to_gdrive:
            from utils.drive_helper import copy_to_gdrive

            if not os.path.exists(g_drive_path):
                if not os.path.exists('/content/drive/My Drive/'):
                    raise NotADirectoryError('Drive not mounted')
                os.makedirs(g_drive_path)

            checkpoint_path = os.path.join(g_drive_path,'pggan_checkpoints.zip')
            logs_path = os.path.join(g_drive_path,'pggan_logs.zip')

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
            extract_data_g_drive('CML/pggan_checkpoints.zip', mounted=True, extracting_checkpoints=True)
            print('Extracted checkpoints from colab')

        # Load the temp configuration
        # Find latest scale file
        # scale = 0
        # for scale in range(self.modelConfig.n_scales-1,-1,-1):
        #     print(scale)
        #     path = os.path.join(self.checkpoint_dir, f'{self.model_label}_{scale}_' + "_tmp_config.json")
        #     print(path)
        #     if os.path.exists(path):
        #         self.temp_config_path = path
        #         break

        # with open(self.temp_config_path,'rb') as infile:
        #     tmpConfig = json.load(infile)
        # self.startScale = tmpConfig["scale"]
        # self.startIter = tmpConfig["iter"]

        # Read the training configuration
        # with open(self.train_config_path,'rb') as infile:
        #     trainConfig = json.load(infile)
        # self.readTrainConfig(trainConfig)

        # Re-initialize the model
        # self.initModel(depthOtherScales = [self.modelConfig.depthScales[i] for i in range(0, self.startScale)])
        
        # Load saved checkpoint
        # if self.transition:
            # self.checkpoint_manager.directory = '{0:}x{0:}'.format(self.cfg.resolution//2)+ '_transition'
        
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")

        # if self.transition:
            # self.checkpoint_manager.directory = self.save_dir

    def train_resolution(self, restore=False, colab=False, load_from_g_drive=False, verbose=True, g_drive_path = '/content/drive/My Drive/CML'):
        """ Train the model. """
        self.colab = colab
        self.train_start_time = time.time()
        self.g_drive_path = g_drive_path
        
        # Create new directories for individual scales/transition
        save_tag = '{0:}x{0:}'.format(self.cfg.resolution)
        if self.transition:
            save_tag += '_transition'
        
        self.img_save_dir = os.path.join(self.image_save_dir, save_tag)
        if not os.path.exists(self.img_save_dir):
            os.makedirs(self.img_save_dir)
        
        self.save_dir = os.path.join(self.checkpoint_dir, save_tag)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Initialise Checkpoints
        self.save_period = self.cfg.save_period
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.Generator,
            discriminator=self.Discriminator
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.save_dir, max_to_keep=3)

        if self.cfg.load_model:
            self.load_saved_training(load_from_g_drive=load_from_g_drive)
        elif self.transition:
            pass
            #TODO load state from previous resolution

        # with tf.device("/cpu:0"):
            # image_batch = image_loader.create_batch_pipeline()

        # self.make_train_op(image_batch)

        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(os.path.join(self.cfg.summary_dir, time.strftime('%Y%m%d_%H%M%S')))

        # Create ops in graph before Session is created
        # init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        # with tf.Session() as sess:
            # sess.run(init)
            # tf.train.start_queue_runners(sess)
            # load_model = self.cfg.load_model
            # if self.cfg.load_model:
                # self.load(sess, saver, load_model)
            # elif transition:
            #     vars_to_load = []
            #     all_vars = tf.trainable_variables()
            #     r = self.cfg.min_resolution
            #     while r < self.cfg.resolution:
            #         var_scope = '{0:}x{0:}'.format(r)
            #         vars_to_load += [v for v in all_vars if var_scope in v.name]
            #         r *= 2
            #     saver_restore = tf.train.Saver(vars_to_load)
            #     tag = '{0:}x{0:}'.format(self.cfg.resolution // 2)
            #     print(tag)
            #     self.load(sess, saver_restore, tag=tag)

            # alpha = self.cfg.fade_alpha
        self.global_step = 0
        # sum_g_loss, sum_d_loss = 0., 0.
        # batch_gen = image_loader.batch_generator()

        while True:
            print(f'Current step: {self.global_step}/{self.n_iters}')
            for real_image_batch in self.train_dataset:
                self.global_step += 1
                self.checkpoint.step.assign_add(1)
                
                # real_image_batch = self.resize_image(real_image_batch, verbose=verbose)
                noise = tf.random.normal([self.batch_size, self.z_dim])

                # batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
                # feed_dict = {self.tf_placeholders['z']: batch_z,
                                # self.tf_placeholders['learning_rate']: learning_rate,
                                # self.tf_placeholders['alpha']: alpha}
                
                self.train_step(real_image_batch, noise, verbose=verbose)

                # if global_step % display_period == 0:
                #     _, global_step, d_loss, merged_res = \
                #         sess.run([self.d_train_op, self.global_step, self.d_loss, merged],
                #                     feed_dict=feed_dict)
                # else:
                #     _, global_step, d_loss = \
                #         sess.run([self.d_train_op, self.global_step, self.d_loss],
                #                 feed_dict=feed_dict)

                # g_loss = 0.
                # if global_step % n_critic == 0:
                #     _, _, g_loss = \
                #         sess.run([self.g_train_op, self.ema_op, self.g_loss],
                #                     feed_dict=feed_dict)
                # sum_g_loss += g_loss
                # sum_d_loss += d_loss

                # Change Alpha
                if self.transition:
                    alpha_step = 1. / self.n_iters
                    self.alpha = min(1., self.cfg.fade_alpha+self.global_step*alpha_step)

                # Write logged losses
                if self.global_step % self.display_period == 0:
                    
                    with self.gen_summary_writer.as_default():
                        # tf.summary.scalar('generator_wasserstein_loss', self.metrics['generator_wasserstein_loss'].result(), step=self.overall_steps)
                        tf.summary.scalar('generator_loss', self.metrics['generator_loss'].result(), step=self.global_step)

                    with self.dis_summary_writer.as_default():
                        # tf.summary.scalar('discriminator_wasserstein_loss_real', self.metrics['discriminator_wasserstein_loss_real'].result(), step=self.overall_steps)
                        tf.summary.scalar('discriminator_wasserstein_loss', self.metrics['discriminator_wasserstein_loss'].result(), step=self.global_step)
                        tf.summary.scalar('discriminator_wasserstein_gradient_penalty', self.metrics['discriminator_wasserstein_gradient_penalty'].result(), step=self.global_step)
                        tf.summary.scalar('discriminator_epsilon_loss', self.metrics['discriminator_epsilon_loss'].result(), step=self.global_step)
                        tf.summary.scalar('discriminator_loss', self.metrics['discriminator_loss'].result(), step=self.global_step)

                    # Save Images
                    generated_images = self.Generator(noise, training=False)
                    # generated_images = generated_images[:, :, :, :] * 0.5 + 0.5
                    # print('Max ',generated_images.numpy().max())
                    # print('min ',generated_images.numpy().min())
                    generated_images_grid = self.image_loader.make_grid(generated_images.numpy())
                    # print('Max ',generated_images_grid.max())
                    # print('min ',generated_images_grid.min())
                    with self.gen_summary_writer.as_default():
                        tf.summary.image('Generated Images', tf.expand_dims(generated_images_grid, axis=0), max_outputs=1, step=self.global_step)

                    # Take a look at real images
                    # real_image_batch = real_image_batch[:, :, :, :] * 0.5 + 0.5
                    real_images_grid = self.image_loader.make_grid(real_image_batch.numpy())
                    with self.dis_summary_writer.as_default():
                        tf.summary.image('Real Images', tf.expand_dims(real_images_grid, axis=0), max_outputs=1, step=self.global_step)

                    plt.figure(figsize=(10, 10))
                    filename = os.path.join(self.img_save_dir, str(self.global_step) + '_fakes_.png')
                    plt.imsave(filename, generated_images_grid)
                    plt.close()

                    # plt.figure(figsize=(10, 10))
                    # filename = os.path.join(self.img_save_dir, str(self.global_step) + '_reals_.png')
                    # plt.imsave(filename, real_images_grid)
                    # plt.close()

                    if verbose:
                        print('Saved images to ', self.img_save_dir)

                    # Reset Losses
                    for k in self.metrics:
                        self.metrics[k].reset_states()

                    if self.transition:
                        print("Using alpha = ", self.alpha)

                # Save Checkpoint
                if self.global_step % self.save_period == 0:
                    self.save_check_point(self.cfg.resolution, self.global_step, verbose=True, save_to_gdrive=self.colab, g_drive_path = self.g_drive_path)

                


                    # writer.add_summary(merged_res, global_step)
                    # print("After {} iterations".format(global_step),
                    #         "Discriminator loss : {:3.5f}  "
                    #         .format(sum_d_loss / display_period),
                    #         "Generator loss : {:3.5f}"
                    #         .format(sum_g_loss / display_period * n_critic))
                    # sum_g_loss, sum_d_loss = 0., 0.

                if self.global_step > self.n_iters:
                    break
            else:
                continue
            break

        #     if global_step % save_period == 0:
        #         print("Saving model in {}".format(save_dir))
        #         saver.save(sess, save_dir, global_step)
        #         if self.cfg.save_images:
        #             gen_images = self.generate_images(save_tag, alpha=alpha)
        #             plt.figure(figsize=(10, 10))
        #             grid = image_loader.grid_batch_images(gen_images)
        #             filename = os.path.join(img_save_dir, str(global_step) + '.png')
        #             plt.imsave(filename, grid)
        # print("Saving model in {}".format(save_dir))
        # saver.save(sess, save_dir, global_step)

    def train_res(self):
        """ Train the model. """
        batch_size = self.cfg.batch_size
        n_iters = self.cfg.n_iters
        n_critic = self.cfg.n_critic
        z_dim = self.cfg.z_dim
        learning_rate = self.cfg.learning_rate
        display_period = self.cfg.display_period
        save_period = self.cfg.save_period
        image_loader = self.image_loader
        transition = self.cfg.transition
        
        # Create new directories for individual scales/transition
        save_tag = '{0:}x{0:}'.format(self.cfg.resolution)
        if transition:
            save_tag += '_transition'
        
        img_save_dir = os.path.join(self.image_save_dir, save_tag)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        
        save_dir = os.path.join(self.checkpoint_dir, save_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, 'model')

        with tf.device("/cpu:0"):
            image_batch = image_loader.create_batch_pipeline()

        self.make_train_op(image_batch)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(self.cfg.summary_dir, time.strftime('%Y%m%d_%H%M%S')))

        # Create ops in graph before Session is created
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            tf.train.start_queue_runners(sess)
            load_model = self.cfg.load_model
            if self.cfg.load_model:
                self.load(sess, saver, load_model)
            elif transition:
                vars_to_load = []
                all_vars = tf.trainable_variables()
                r = self.cfg.min_resolution
                while r < self.cfg.resolution:
                    var_scope = '{0:}x{0:}'.format(r)
                    vars_to_load += [v for v in all_vars if var_scope in v.name]
                    r *= 2
                saver_restore = tf.train.Saver(vars_to_load)
                tag = '{0:}x{0:}'.format(self.cfg.resolution // 2)
                print(tag)
                self.load(sess, saver_restore, tag=tag)

            alpha = self.cfg.fade_alpha
            global_step = 0
            sum_g_loss, sum_d_loss = 0., 0.
            # batch_gen = image_loader.batch_generator()

            for i in range(self.cfg.n_iters):
                batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
                feed_dict = {self.tf_placeholders['z']: batch_z,
                             self.tf_placeholders['learning_rate']: learning_rate,
                             self.tf_placeholders['alpha']: alpha}
                if global_step % display_period == 0:
                    _, global_step, d_loss, merged_res = \
                        sess.run([self.d_train_op, self.global_step, self.d_loss, merged],
                                 feed_dict=feed_dict)
                else:
                    _, global_step, d_loss = \
                        sess.run([self.d_train_op, self.global_step, self.d_loss],
                             feed_dict=feed_dict)

                g_loss = 0.
                if global_step % n_critic == 0:
                    _, _, g_loss = \
                        sess.run([self.g_train_op, self.ema_op, self.g_loss],
                                 feed_dict=feed_dict)
                sum_g_loss += g_loss
                sum_d_loss += d_loss
                if transition:
                    alpha_step = 1. / n_iters
                    alpha = min(1., self.cfg.fade_alpha+global_step*alpha_step)
                if global_step % display_period == 0:
                    writer.add_summary(merged_res, global_step)
                    print("After {} iterations".format(global_step),
                          "Discriminator loss : {:3.5f}  "
                          .format(sum_d_loss / display_period),
                          "Generator loss : {:3.5f}"
                          .format(sum_g_loss / display_period * n_critic))
                    sum_g_loss, sum_d_loss = 0., 0.
                    if transition:
                        print("Using alpha = ", alpha)
                if global_step % save_period == 0:
                    print("Saving model in {}".format(save_dir))
                    saver.save(sess, save_dir, global_step)
                    if self.cfg.save_images:
                        gen_images = self.generate_images(save_tag, alpha=alpha)
                        plt.figure(figsize=(10, 10))
                        grid = image_loader.grid_batch_images(gen_images)
                        filename = os.path.join(img_save_dir, str(global_step) + '.png')
                        plt.imsave(filename, grid)
            print("Saving model in {}".format(save_dir))
            saver.save(sess, save_dir, global_step)

    # def generate_images(self, model, batch_z=None, alpha=0.):
    #     """Runs generator to generate images"""
    #     batch_size = 64  # self.cfg.batch_size
    #     z_dim = self.cfg.z_dim
    #     if batch_z is None:
    #         batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
    #     # saver = tf.train.Saver(self.ema_vars)
    #     saver = tf.train.Saver()
    #     feed_dict = {self.tf_placeholders['z']: batch_z,
    #                  self.tf_placeholders['alpha']: alpha}
    #     image_loader = self.image_loader
    #     gen = self.build_generator(training=False)

    #     with tf.Session() as sess:
    #         self.load(sess, saver, model)
    #         gen_images = sess.run(gen, feed_dict=feed_dict)
    #         gen_images = image_loader.postprocess_image(gen_images)
    #         return gen_images

    def load(self, sess, saver, tag=None):
        """ Load the trained model. """
        if tag is None:
            tag = '{0:}x{0:}'.format(self.cfg.input_shape[0])

        load_dir = os.path.join(self.cfg.model_save_dir, tag, 'model')
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(load_dir))
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        # print(checkpoint.model_checkpoint_path)
        saver.restore(sess, os.path.join(self.cfg.model_save_dir,self.cfg.load_model,os.path.split(checkpoint.model_checkpoint_path)[-1]))
