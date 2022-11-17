import os
# Do not show all the mesagges genrated by tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
# from IPython import display
from tqdm import tqdm
import numpy as np

from dataloader_pipeline import Dataloader
from models import Models
from typing import Tuple, Type

import logging

import math 
from modules import summary_tensorboard, pearson_r

class ARAGAN(object):
    def __init__(self):
        # Buffer size, complete training set length 
        self.BUFFER_SIZE = 98723
        
        multiplier_dgx = 2
        self.BATCH_SIZE = 32 * multiplier_dgx
        # Each image is 256x256 in size
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256

        #  GENERATOR
        self.OUTPUT_CHANNELS = 3
        # Lamda parameter to calculate the loss function, this parameter will 
        # set the weight of the L1 loss
        self.LAMBDA = 100

        # Training epochs
        self.EPOCHS = 100
        # Number of images in both sets
        self.TOTAL_IMGS = 98723
        self.TOTAL_IMGS_TEST = 32196
        
        initial_learning_rate=1e-4 * math.sqrt(multiplier_dgx)

        # Cublass error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Call the Models class setting some parameters 
        self.models = Models(self.IMG_WIDTH, 
                             self.IMG_HEIGHT,
                             self.OUTPUT_CHANNELS,
                             self.BATCH_SIZE)
        
        # Search the models available to be choose by the developer 
        method_list = [method for method in dir(Models) 
                       if method.startswith('__') is False]
        print('\033[1;32mModels available: \033[0;0m', method_list)
        # Choose the Generator architecture from the list
        self.name = input('Choose the Generator from the list above: ')
        
        # Call the Generator and the Discriminator
        self.generator = eval('self.models.' + self.name + '()')
        self.discriminator = self.models.Discriminator()
        self.discriminator.summary()
        
        
        self.name += '_' + input("Hyperparameters: ")
        
        # Create the BCE loss
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Create the KLD metric
        self.auc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')
        self.kld = tf.keras.metrics.KLDivergence(
            name='kullback_leibler_divergence', dtype=None)
        
        # Create the schelue for the learning rate
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.9)
        
        # Create the optimizer for the generator and the discriminator 
        self.generator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate,
            beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, 
            beta_1=0.5)
        
        # Set logs folder, tensorboard writer
        self.log_dir="logs/"
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + 
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 
            "_" + self.name + "_" + str(self.BATCH_SIZE))
        
        # Set checkpoints folder 
        self.checkpoint_dir = './training_checkpoints'
        self.chekpoint_name =  self.name + "_"+ str(self.BATCH_SIZE) + "/epoch_"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 
                                              self.chekpoint_name)
        
        # Compile models
        self.generator.compile(optimizer = self.generator_optimizer,
                           loss = self.generator_loss)  
        self.discriminator.compile(optimizer = self.discriminator_optimizer,
                           loss = self.discriminator_loss) 
        
        # logging_file = os.path.join('training_checkpoints',
        #                             self.name + "_"+ str(self.BATCH_SIZE),
        #                             'logging_file.log')
        logging_file = ''.join(('training_checkpoints/',
                                self.name + "_"+ str(self.BATCH_SIZE),
                                '/logging_file.log'))
        os.makedirs(''.join(('training_checkpoints/',
                                self.name + "_"+ str(self.BATCH_SIZE))))
        logging.basicConfig(filename=logging_file, 
                            filemode='w', 
                            level=logging.INFO, 
                            force=True)
               
    def generator_loss(self, 
                       disc_generated_output: tf.Tensor,
                       gen_output: tf.Tensor, 
                       target: tf.Tensor) -> Tuple[tf.Tensor, 
                                                   tf.Tensor, 
                                                   tf.Tensor]:
        ''' Generator loss 
        BCE -> Binary CrossEntropy
        BCE(ones, discriminator output) + lamda * L1_loss(GT, generator output)

        Args:
            disc_generated_output (tf.Tensor): output generated by the 
            discriminator
            gen_output (tf.Tensor): output generated by the generator 
            target (tf.Tensor): ground truth from the dataset 

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: total_gen_loss, gan_loss, 
            l1_loss
        '''
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), 
                                    disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self,
                           disc_real_output: tf.Tensor, 
                           disc_generated_output: tf.Tensor) -> tf.Tensor:
        '''Discriminator loss
        BCE(ones, )

        Args:
            disc_real_output (tf.Tensor): discriminator output of the GT image
            disc_generated_output (tf.Tensor): discriminator output of the 
            generated image

        Returns:
            tf.Tensor: discriminator loss
        '''
        real_loss = self.loss_object(tf.ones_like(disc_real_output),
                                     disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output),
                                          disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss 
    
    def calculate_metrics(self,
                          target: tf.Tensor, 
                          gen_output: tf.Tensor) -> Tuple[tf.Tensor, 
                                                          tf.Tensor, 
                                                          tf.Tensor]:
        '''Function to calculate model metrics:
        Kullback-Leibler Divergence (KLD)
        Pearson’s Correlation Coefficient, CC
        Shuffled Area Under the ROC Curve (s-AUC)
        
        Args:
            target (tf.Tensor): attention map from the GT
            gen_output (tf.Tensor): attention map generated by the generator

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: output metrics
        '''

        
        kld_metric =  self.kld(target, gen_output)
        mae_metric =  tf.keras.metrics.mean_absolute_error(target, gen_output)
        mse_metric =  tf.keras.metrics.mean_squared_error(target, gen_output)

        kld_metric = tf.reduce_mean(kld_metric)
        mae_metric = tf.reduce_mean(mae_metric)
        mse_metric = tf.reduce_mean(mse_metric)

        correlation_coefficient = pearson_r(target, gen_output)
        
        # correlation_coefficient = self.cross_entropy(target, gen_output)
        auc = self.auc(target, gen_output)

        # s_auc = np.std(auc)

        # print(kld_metric.numpy())
        
        return  kld_metric, mae_metric, mse_metric, correlation_coefficient, auc

    shape_img = (None, 256, 256 ,3)
    signature_inputs = [
        tf.TensorSpec(shape=shape_img, dtype=tf.float32),
        tf.TensorSpec(shape=shape_img, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64)]
    
    @tf.function(input_signature=signature_inputs)
    def train_step(self,
                   input_image: tf.Tensor,
                   target: tf.Tensor,
                   step: int,
                   epoch: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''Train step for every batch of the training dataset

        Args:
            input_image (tf.Tensor): RGB image
            target (tf.Tensor): Attention map from th GT
            step (int): train step
            epoch (int): Train epoch

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: losses to be evaluated later
        '''
        # Create variables for memory optimization
        disc_real_output = None
        disc_gen_output = None
        gen_total_loss = None
        gen_gan_loss = None
        gen_l1_loss = None
        disc_loss = None

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Forward pass
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target],
                                                  training=True)
            disc_gen_output = self.discriminator([input_image, gen_output],
                                                 training=True)

            # Generator Loss value for this batch
            gen_loss, total_loss_gan, gen_l1_loss = self.generator_loss(
                disc_gen_output, 
                gen_output, 
                target)
            # Discriminator Loss value for this batch
            disc_loss = self.discriminator_loss(disc_real_output, 
                                                disc_gen_output)

        # Get generator gradients of loss wrt the weights
        generator_gradients = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables)
        # Get discriminator gradients of loss wrt the weights
        discriminator_gradients = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables)

        # Update the weights of the generator
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, 
                self.generator.trainable_variables))
        # Update the weights of the discriminator
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables))

        # Calculate metrics for this batch
        kld_metric, mae_metric, mse_metric, correlation_coefficient, auc \
            = self.calculate_metrics(target, gen_output)

        # Write logs for Tensorboard 
        tensorboard_step = step * self.BATCH_SIZE + epoch * self.TOTAL_IMGS
        
        summary_tensorboard(self.summary_writer,
                            'Train', 
                            tensorboard_step,
                            gen_loss,
                            disc_loss,
                            total_loss_gan,
                            kld_metric,
                            mae_metric,
                            mse_metric, 
                            correlation_coefficient, 
                            auc,
                            self.generator_optimizer.learning_rate)

        return gen_loss, total_loss_gan, disc_loss

    def fit(self, train_ds: tf.data.Dataset, epoch: int) -> None:
        '''Train function to pass the batchs from the dataset to the train step
        procedure

        Args:
            train_ds (tf.data.Dataset): Training dataset
            test_ds (tf.data.Dataset): Testing dataset
        '''
        gen_total_losses = []
        gen_gan_losses = []
        disc_losses = []
        # Iterate the training dataset in batches 
        str_tqdm = ''.join(('\033[1;34mTraining epoch \033[0;0m', 
                        str(epoch), '/', str(self.EPOCHS)))
        for step, (input_image, target) in tqdm(train_ds.enumerate(),
                desc = str_tqdm,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                dynamic_ncols  = True):
            gen_total_loss, gen_gan_loss, disc_loss = self.train_step(
                input_image, target, step, epoch)
            # Store losses 
            gen_total_losses.append(gen_total_loss)
            gen_gan_losses.append(gen_gan_loss)
            disc_losses.append(disc_loss)
          
        print("gen_total_loss {:1.2f}".format(np.mean(gen_total_losses)))
        print("gen_gan_loss {:1.2f}".format(np.mean(gen_gan_losses)))  
        print("disc_loss {:1.2f}".format(np.mean(disc_losses)))
        logging.info("gen_total_loss {:1.2f}".format(np.mean(gen_total_losses)))
        logging.info("gen_gan_loss {:1.2f}".format(np.mean(gen_gan_losses)))
        logging.info("disc_loss {:1.2f}".format(np.mean(disc_losses)))                                                
                                                        
    @tf.function(input_signature=signature_inputs)
    def test_step(self, 
                  input_image: tf.Tensor, 
                  target: tf.Tensor, 
                  step: int,
                  epoch: int) -> None:
        '''Test step for every batch of the testing dataset

        Args:
            input_image (tf.Tensor): RGB image
            target (tf.Tensor): Attention map from th GT
            step (int): train step
            epoch (int): Train epoch
        '''
        # Test Forward
        gen_output = self.generator(input_image, training=False)
        
        disc_real_output = self.discriminator([input_image, target],
                                              training=True)
        disc_gen_output = self.discriminator([input_image, gen_output],
                                             training=True)

        # Generator Loss value for this batch
        gen_loss, total_loss_gan, gen_l1_loss = self.generator_loss(
            disc_gen_output, 
            gen_output, 
            target)
        # Discriminator Loss value for this batch
        disc_loss = self.discriminator_loss(disc_real_output, 
                                            disc_gen_output)

        # Calculate metrics for this batch
        kld_metric, mae_metric, mse_metric, correlation_coefficient, auc \
            = self.calculate_metrics(target, gen_output)

        # Write logs for Tensorboard 
        tensorboard_step = step * self.BATCH_SIZE + epoch * self.TOTAL_IMGS_TEST
        
        summary_tensorboard(self.summary_writer,
                            'Test', 
                            tensorboard_step,
                            gen_loss,
                            disc_loss,
                            total_loss_gan,
                            kld_metric,
                            mae_metric,
                            mse_metric, 
                            correlation_coefficient, 
                            auc,
                            self.generator_optimizer.learning_rate)
            
        return kld_metric, mae_metric, mse_metric, correlation_coefficient, auc
                      
    def test(self, test_ds: tf.data.Dataset, epoch: int) -> None:
        '''Test function to pass the batchs from the dataset to the train step
        procedure

        Args:
            test_ds (tf.data.Dataset): Testing dataset
            epoch (int): Epoch
        '''
        # Iterate the test dataset in batches for testing
        str_tqdm = ''.join(('\033[1;34mTesting epoch \033[0;0m', 
                            str(epoch), '/', str(self.EPOCHS)))
        kld_metric_list, mae_metric_list, mse_metric_list, \
            correlation_coefficient_list, auc_list = [], [], [], [], []
        for step, (input_image, target) in tqdm(test_ds.enumerate(),
                 desc = str_tqdm,
                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                 dynamic_ncols  = True):
            kld_metric, mae_metric, mse_metric, correlation_coefficient, auc \
                = self.test_step(input_image, target, step, epoch)
                
            kld_metric_list.append(kld_metric)
            mae_metric_list.append(mae_metric)
            mse_metric_list.append(mse_metric)
            correlation_coefficient_list.append(correlation_coefficient)
            auc_list.append(auc)
            
        print("KLD {:1.2f}".format(np.mean(kld_metric_list)))
        print("MAE {:1.2f}".format(np.mean(mae_metric_list)))  
        print("MSE {:1.2f}".format(np.mean(mse_metric_list)))
        print("CC {:1.2f}".format(np.mean(correlation_coefficient_list)))
        print("AUC {:1.2f}".format(np.mean(auc_list)))
        
        logging.info("KLD {:1.2f}".format(np.mean(kld_metric_list)))
        logging.info("MAE {:1.2f}".format(np.mean(mae_metric_list)))
        logging.info("MSE {:1.2f}".format(np.mean(mse_metric_list)))
        logging.info("CC {:1.2f}".format(np.mean(correlation_coefficient_list)))
        logging.info("AUC {:1.2f}".format(np.mean(auc_list)))

        return np.mean(kld_metric_list), np.mean(mae_metric_list), \
            np.mean(mse_metric_list) ,np.mean(correlation_coefficient_list), \
            np.mean(auc_list)
    
    def dataset_pipeline(self, 
                         dataloader: Dataloader, 
                         image_path : str, 
                         dataset_split : str) -> tf.data.Dataset:
        '''Dataloader pipeline to create training and testing dataset

        Args:
            dataloader (Dataloader): dataloader class 
        '''
        # List all the RGB images in the training dataset to create the Dataset
        dataset = tf.data.Dataset.list_files(image_path)
        
        if dataset_split == 'train':                
            # Shuffle the images, do this before mapping to use the maximun buffer 
            dataset = dataset.shuffle(self.BUFFER_SIZE)
            dataset = dataset.map(dataloader.load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)  
        else: 
            dataset = dataset.map(dataloader.load_image_test)
        # Get the RGB images and the attention map from the dataset
        
        # Create batches with the predefined batch size   
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder = True)

        return dataset

    def main(self) -> None:
        '''Main funtion
        '''
        
        dataloader = Dataloader(aragan.IMG_WIDTH, 
                            aragan.IMG_HEIGHT, 
                            aragan.OUTPUT_CHANNELS)
        PATH = 'dataset/BDDA/'
        train_path = str(PATH + 'training/camera_images/all_images/*.jpg')
        train_dataset = self.dataset_pipeline(dataloader, train_path, 'train')
        
        PATH = 'dataset/BDDA/'
        test_path = str(PATH + 'test/camera_images/all_images/*.jpg')
        test_dataset = self.dataset_pipeline(dataloader, test_path, 'test')
        
        last_test_metric = 0
    
        for epoch in range(self.EPOCHS):
            self.fit(train_dataset, epoch)            
            kld_metric, mae_metric, mse_metric, correlation_coefficient, auc \
                 = self.test(test_dataset, epoch)
                 
                 
            # Save generator for inference
            # if test_metric > last_test_metric:
            #     last_test_metric = test_metric
                
            # print('checkpoint_prefix: ',self.checkpoint_dir)
            # if os.path.exists(self.checkpoint_dir):
            #     test = os.listdir(self.checkpoint_dir)
            #     for item in test:
            #         if item.endswith(".h5"):
            #             os.remove(
            #                 os.path.join(self.checkpoint_dir, item))
            checkpoint = ''.join((self.checkpoint_prefix, 
                                str(epoch), '.h5'))
            print('Checkpoint: ',checkpoint)
            logging.info('Saving checkpoint for epoch {:1.3f}, \
                with kld_metric {:1.3f}, mae_metric {:1.3f}, mse_metric {:1.3f},\
                    correlation_coefficient {:1.3f}, auc {:1.3f},'.format(
                    epoch+1, kld_metric, mae_metric, 
                    mse_metric, correlation_coefficient, auc))
            print ('Saving checkpoint for epoch {:1.3f}, \
                with kld_metric {:1.3f}, mae_metric {:1.3f}, mse_metric {:1.3f},\
                    correlation_coefficient {:1.3f}, auc {:1.3f},'.format(
                    epoch+1, kld_metric, mae_metric, 
                    mse_metric, correlation_coefficient, auc))
            
            self.generator.save(checkpoint)

if __name__ == "__main__":
    aragan = ARAGAN()
    aragan.main() 