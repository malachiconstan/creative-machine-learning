import time
import os
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from PIL import Image
from IPython import display

from utils.configs.pggan_config import _C
from utils.config import BaseConfig, getConfigFromDict
from utils.models import ProgressiveGAN
from utils.preprocessing import get_image_dataset
from utils.losses import WGANGPGradientPenalty

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
                 miniBatchScheduler=None,
                #  datasetProfile=None,
                 configScheduler=None,
                #  useGPU=True,
                #  visualisation=None,
                 lossIterEvaluation=200,
                 saveIter=5000,
                 checkPointDir=None,
                 modelLabel="GAN",
                 config=None,
                 pathAttribDict=None,
                 selectedAttributes=None,
                 imagefolderDataset=False,
                 ignoreAttribs=False):
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

        self.configScheduler = {}
        if configScheduler is not None:
            self.configScheduler = {
                int(key): value for key, value in configScheduler.items()}

        self.miniBatchScheduler = {}
        if miniBatchScheduler is not None:
            self.miniBatchScheduler = {
                int(x): value for x, value in miniBatchScheduler.items()}

        # self.datasetProfile = {}
        # if datasetProfile is not None:
        #     self.datasetProfile = {
        #         int(x): value for x, value in datasetProfile.items()}

        self.datapath = datapath
        
        if config is None:
            config = {}

        self.readTrainConfig(config)

        # Intern state
        self.runningLoss = {}
        self.startScale = 0
        self.startIter = 0
        self.lossProfile = []

        self.initModel()

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        # Checkpoints ?
        self.checkPointDir = checkPointDir
        self.modelLabel = modelLabel
        self.saveIter = saveIter
        self.pathLossLog = None


        if self.checkPointDir is not None:
            self.pathLossLog = os.path.abspath(os.path.join(self.checkPointDir,
                                                            self.modelLabel
                                                            + '_losses.pkl'))
            self.pathRefVector = os.path.abspath(os.path.join(self.checkPointDir,
                                                              self.modelLabel
                                                              + '_refVectors.pt'))

        # Loss printing
        self.lossIterEvaluation = lossIterEvaluation
        
    def initModel(self):
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
            lambdaGP = self.modelConfig.lambdaGP
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
            low_res_real = F.avg_pool2d(input_real, (2, 2))
            low_res_real = F.upsample(
                low_res_real, scale_factor=2, mode='nearest')

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
    
    def saveBaseConfig(self, outPath):
        """
        Save the model basic configuration (the part that doesn't change with
        the training's progression) at the given path
        """

        outConfig = getDictFromConfig(
            self.modelConfig, self.getDefaultConfig())

        if "alphaJumpMode" in outConfig:
            if outConfig["alphaJumpMode"] == "linear":

                outConfig.pop("iterAlphaJump", None)
                outConfig.pop("alphaJumpVals", None)

        with open(outPath, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

    def train(self):
        """
        Launch the training. This one will stop if a divergent behavior is
        detected.
        Returns:
            - True if the training completed
            - False if the training was interrupted due to a divergent behavior
        """
        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                          + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)

        for scale in range(self.startScale, self.modelConfig.n_scales):
            
            # Get train dataset at the correct image scale
            train_dataset = get_image_dataset(self.datapath,
                                            img_height=self.modelConfig.size_scales[scale],
                                            img_width=self.modelConfig.size_scales[scale],
                                            batch_size=self.modelConfig.miniBatchSize,
                                            normalize=True)
            
            # Get number of batches in the train dataset
            number_batches = len(train_dataset)

            shiftIter = 0
            if self.startIter > 0:
                shiftIter = self.startIter
                self.startIter = 0

            shiftAlpha = 0

            # While the shiftAlpha variable is less than the jumps of alpha in that scale and the iteration which the shiftAlpha corresponds to in that scale is less than the shiftIter, add 1 to shiftAlpha
            # Basically this tells us what is the level of alpha we should start at (the one right before the shiftIter)
            while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
                shiftAlpha += 1

            while shiftIter < self.modelConfig.maxIterAtScale[scale]:
                
                # Set the index to set alpha to to the current shiftAlpha
                self.indexJumpAlpha = shiftAlpha
                
                status = self.train_epoch(train_dataset,
                                            scale,
                                            shiftIter=shiftIter,
                                            maxIter=self.modelConfig.maxIterAtScale[scale])

                if not status:
                    return False

                shiftIter += number_batches
                
                # Update shiftAlpha to the next step
                while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
                    shiftAlpha += 1

            # Save checkpoint TODO

            # If final scale then don't add anymore layers
            if scale == n_scales - 1:
                break

            # Add scale
            self.model.addScale(self.modelConfig.depthScales[scale + 1])

        self.startScale = n_scales
        self.startIter = self.modelConfig.maxIterAtScale[-1]
        return True


    def train_epoch(self,
                    dataset,
                    scale,
                    shiftIter=0,
                    maxIter=-1):
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

        i = shiftIter

        for image_batch in dataset:
            
            inputs_real = image_batch
            # inputs_real = data[0]
            # labels = data[1]

            if inputs_real.shape[0] < self.modelConfig.miniBatchSize:
                raise ValueError('Image batch shape less than mini_batch_size')

            # Additionnal updates inside a scale
            inputs_real = self.inScaleUpdate(i, scale, inputs_real)

            i += 1

            # Regular evaluation
            if i % self.lossIterEvaluation == 0:

                # Reinitialize the losses
                self.updateLossProfile(i)

                print('[%d : %6d] loss G : %.3f loss D : %.3f' % (scale, i,
                      self.lossProfile[-1]["lossG"][-1],
                      self.lossProfile[-1]["lossD"][-1]))

                self.resetRunningLosses()

                if self.visualisation is not None:
                    self.sendToVisualization(inputs_real, scale)

            if self.checkPointDir is not None:
                if i % self.saveIter == 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, i))
                    self.saveCheckpoint(self.checkPointDir,
                                        labelSave, scale, i)

            if i == maxIter:
                return True

        return True
    
    @tf.function
    def train_step(self, images):
        # Generate noise image
        noise = tf.random.normal([images.shape[0], self.modelConfig.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # 1. Real Output + Wasserstein Loss
            real_output = self.model.netD(images, training=True)
            dnet_wloss_real = self.model.loss_criterion.getCriterion(real_output, True)

            # 2. Fake Output + Wasserstein Loss
            generated_images = self.model.netG(noise, training=True)
            fake_output = self.model.netD(generated_images, training=True)
            dnet_wloss_fake = self.model.loss_criterion.getCriterion(fake_output, False)
            gnet_wloss_fake = self.model.loss_criterion.getCriterion(fake_output, True)

            # 3. Wasserstein Gradient Loss
            # dnet_wgrad_loss = WGANGPGradientPenalty() TODO

            

            # sdis_loss(disc_loss)
            # sgen_loss(gen_loss)
            # sdis_acc(tf.ones_like(real_output), real_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))