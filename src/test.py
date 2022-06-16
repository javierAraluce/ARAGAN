import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import time
from tqdm import tqdm
import datetime
from dataloader_pipeline import Dataloader
import os
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import csv
from matplotlib.gridspec import SubplotSpec

from PIL import Image, ImageFont, ImageDraw
import imageio


    
class TestModel(object):
    def __init__(self):
        # Cublass error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # tf.enable_eager_execution(True)
        self.TOTAL_IMGS_TEST = 32196
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.OUTPUT_CHANNELS = 3
        self.PATH_dataset = 'dataset/BDDA/'
        self.PATH_dataset_DADA = 'dataset/DADA2000/'
        self.checkpoint_dir = '../AraNet/training_checkpoints'

        # print('Models available: {}'.format(
        #     sorted(os.listdir(self.checkpoint_dir))))


        # self.name = input("Select model? ")
        self.name = 'MultiHead_conv_output_22M_8'
        self.BATCH_SIZE = 1 #int(self.name.split('_')[-1])
        
        self.dataloader = Dataloader(self.IMG_WIDTH, 
                                     self.IMG_HEIGHT, 
                                     self.OUTPUT_CHANNELS)

        image_path_test = str(self.PATH_dataset + 
                              'test/camera_images/all_images/*.jpg')
        self.test_dataset = tf.data.Dataset.list_files(image_path_test, seed = 5, shuffle=False)
        self.test_dataset = self.test_dataset.map(
            self.dataloader.load_image_test)
        self.test_dataset = self.test_dataset.batch(self.BATCH_SIZE)

        image_path_test_DADA = str(self.PATH_dataset_DADA + 
                                   'testing/camera_images/all_images/*.jpg')
        self.test_dataset_DADA = tf.data.Dataset.list_files(image_path_test_DADA, seed = 5, shuffle=False)
        self.test_dataset_DADA = self.test_dataset_DADA.map(
            self.dataloader.load_image_test_dada)
        self.test_dataset_DADA = self.test_dataset_DADA.batch(self.BATCH_SIZE)


        self.display_images = False


        self.log_dir="logs/"
        
        self.auc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')
        self.kld = tf.keras.metrics.KLDivergence(
            name='kullback_leibler_divergence', dtype=None)

        # Mean metrics

        self.mean_kld_bdda = 0
        self.mean_cc_bdda = 0
        self.mean_sAUC_bdda = 0

        self.mean_kld_dada = 0
        self.mean_cc_dada = 0
        self.mean_sAUC_dada = 0

        self.list_kld = []
        self.list_cc = []
        self.list_sAUC = []

        self.best_kld = 1
        self.best_cc = 0
        
        self.images_video_list = []
   
    def correlation_coefficient(self, patch1, patch2):

        product = tf.math.reduce_mean(
            (patch1 - tf.math.reduce_mean(patch1)) * 
            (patch2 - tf.math.reduce_mean(patch2))) 
        stds = tf.math.reduce_std(patch1) * tf.math.reduce_std(patch2)
        if stds == 0.0:
            return 0.0
        else:
            product /= stds
            return product

    def calculate_metrics(self, target, gen_output):
        kld_metric =  self.kld(target, gen_output)
        mae_metric =  tf.keras.metrics.mean_absolute_error(target, gen_output)
        mse_metric =  tf.keras.metrics.mean_squared_error(target, gen_output)

        kld_metric = tf.reduce_mean(kld_metric)
        mae_metric = tf.reduce_mean(mae_metric)
        mse_metric = tf.reduce_mean(mse_metric)

        correlation_coefficient = self.pearson_r(target, gen_output)
        
        # correlation_coefficient = self.cross_entropy(target, gen_output)
        auc = self.auc(target, gen_output)

        # s_auc = np.std(auc)

        # print(kld_metric.numpy())
        self.list_kld.append(kld_metric.numpy())
        self.list_cc.append(correlation_coefficient.numpy())
        self.list_sAUC.append(auc.numpy())

        return  kld_metric, mae_metric, mse_metric, correlation_coefficient, auc

    def pearson_r(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = tf.reduce_mean(x, axis=1, keepdims=True)
        my = tf.reduce_mean(y, axis=1, keepdims=True)
        xm, ym = x - mx, y - my
        t1_norm = tf.nn.l2_normalize(xm, axis = 1)
        t2_norm = tf.nn.l2_normalize(ym, axis = 1)
        cosine = tf.compat.v1.losses.cosine_distance(t1_norm, t2_norm, axis = 1)
        return cosine

    # @tf.function
    def test_step(self, input_image, target, step, input_image_raw, dataset = 'bdda'):
        gen_output = self.generator(input_image, training=False)
        

        l1_metric = tf.reduce_mean(tf.abs(target - gen_output))
        kld_metric, mae_metric, mse_metric, correlation_coefficient, auc = \
            self.calculate_metrics(target, gen_output)

        # if self.display_images:
        #     if kld_metric < self.best_kld:
        #         self.best_kld = kld_metric 
                # self.generate_video_images(input_image, 
                #                 target, 
                #                 gen_output, 
                #                 step, 
                #                 dataset, 
                #                 'kld' ,
                #                 kld_metric))
                 
        self.images_video_list.append(self.generate_video_images(
            input_image, 
            target, 
            gen_output, 
            step, 
            dataset, 
            'kld' ,
            kld_metric))

            # if correlation_coefficient > self.best_cc:
            #     self.best_cc = correlation_coefficient  
            #     self.generate_images(input_image, 
            #                          target, 
            #                          gen_output,
            #                          step,
            #                          dataset, 
            #                          'cc' ,
            #                          correlation_coefficient)

        tensorboard_step = step * self.BATCH_SIZE
        with self.summary_writer.as_default():
            tf.summary.scalar('testing_l1_metric_' + dataset,  
                              l1_metric,  
                              step = tensorboard_step)
            tf.summary.scalar('testing_kld_metric_' + dataset, 
                              kld_metric, 
                              step = tensorboard_step)
            tf.summary.scalar('testing_mae_metric_' + dataset,
                              mae_metric, 
                              step = tensorboard_step)
            tf.summary.scalar('testing_mse_metric_' + dataset, 
                              mse_metric, 
                              step = tensorboard_step)
            tf.summary.scalar('testing_correlation_coefficient_' + dataset, 
                              correlation_coefficient,
                              step = tensorboard_step)
            tf.summary.scalar('testing_roc_auc_score_' + dataset,
                              auc, 
                              step = tensorboard_step)

    def create_subtitle(self, fig: plt.Figure, grid: SubplotSpec, title: str):
        "Sign sets of subplots with title"
        row = fig.add_subplot(grid)
        # the '\n' is important
        row.set_title(f'\n{title}\n', fontweight='semibold' , size=50)
        # hide subplot
        row.set_frame_on(False)
        row.axis('off')

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
    
    def generate_video(self, images, step, epoch):    
        # name = 'GIFS_bdda/temp_result_' + str(step) + '_' + str(epoch) + '.gif'        
        # imageio.mimwrite(name, images, 'GIF', duration=0.1)
        
        name = 'videos_bdda/temp_result_' + str(step) + '_' + str(epoch) + '.mp4'     
        writer = imageio.get_writer(name, fps=10)

        for im in images:
            # im is numpy array
            writer.append_data(im)
        writer.close()

    def generate_video_dada(self, images, step, epoch):    
        # name = 'GIFS_dada/temp_result_' + str(step) + '_' + str(epoch) + '.gif'        
        # imageio.mimwrite(name, images, 'GIF', duration=1/30)
        
        name = 'videos_dada/temp_result_' + str(step) + '_' + str(epoch) + '.mp4'     
        writer = imageio.get_writer(name, fps=30)

        for im in images:
            # im is numpy array
            writer.append_data(im)
        writer.close()

        
        
    def generate_video_images(self, 
                        input_image, 
                        target, 
                        gen_output, 
                        step, 
                        dataset, 
                        metric, 
                        value):       
        
        img_size_height = int(720 / 1)
        img_size_width = int(1280 / 1)
        image_comp = Image.new('RGB', (img_size_width + 128, img_size_height * 2), color=0)
        image_text = Image.new('RGB', (128, img_size_height * 2), color=0)
        d = ImageDraw.Draw(image_text)
        font = ImageFont.truetype("times-ro.ttf", 25)
        # d.text((10, 128), "    RGB", font=font, fill=(255, 255, 255, 255))
        d.text((10, (img_size_height / 2) + 0), "    RGB \n\n     GT", 
               font=font, 
               fill=(255, 255, 255, 255))
        d.text((10, (img_size_height / 2) + img_size_height * 1), "    RGB \n\nPrediction", 
               font=font, 
               fill=(255, 255, 255, 255))
        image_comp.paste(image_text, (0, 0))
        
        for j in range(self.BATCH_SIZE):  
            # input_image_resized = input_image  
            input_image_resized = tf.image.resize(input_image[j], 
                                            [img_size_height, img_size_width],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            gen_output_resized = tf.image.resize(gen_output[j], 
                                            [img_size_height, img_size_width],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            target_resized = tf.image.resize(target[j], 
                                            [img_size_height, img_size_width],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            image = self.tensor_to_image(input_image_resized)
            map = self.tensor_to_image(gen_output_resized)
            gt = self.tensor_to_image(target_resized)
            map = Image.blend(image, map, 0.7)
            gt = Image.blend(image, gt, 0.7)
            # image_comp.paste(image, (120, 0))
            image_comp.paste(map, (120, img_size_height * 1))
            image_comp.paste(gt, (120, img_size_height * 0))
        
        image_comp = np.asarray(image_comp)  
        # image_comp.save(
        #     '{}_at_dataset_{}_at step_{:04d}_metric_{}\
        #         _value_{}.png'.format(self.name, dataset, step,metric, value))

        return image_comp
     
    def generate_images(self, 
                        input_image, 
                        target, 
                        gen_output, 
                        step, 
                        dataset, 
                        metric, 
                        value):       
        
        image_comp = Image.new('RGB', (2048 + 120, 256*3), color=0)
        image_text = Image.new('RGB', (120, 256*3), color=0)
        d = ImageDraw.Draw(image_text)
        font = ImageFont.truetype("times-ro.ttf", 25)
        d.text((10, 128), "    RGB", font=font, fill=(255, 255, 255, 255))
        d.text((10, 100 + 256), "    RGB \n\n     GT", 
               font=font, 
               fill=(255, 255, 255, 255))
        d.text((10, 100 + 512), "    RGB \n\nPrediction", 
               font=font, 
               fill=(255, 255, 255, 255))
        image_comp.paste(image_text, (0, 0))
        
        for j in range(self.BATCH_SIZE): 
            image = self.tensor_to_image(input_image[j])
            map = self.tensor_to_image(gen_output[j])
            gt = self.tensor_to_image(target[j])
            im = Image.blend(image, map, 0.7)
            im_gt = Image.blend(image, gt, 0.7)
            image_comp.paste(image, (256 * j + 120, 0))
            image_comp.paste(im, (256 * j + 120, 512))
            image_comp.paste(im_gt, (256 * j + 120, 256))
        
        image_comp.save(
            'img_bbda_inferencia/{}_at_dataset_{}_at step_{:04d}_metric_{}\
                _value_{}.png'.format(self.name, dataset, step,metric, value))

    def main(self):
        
        # for epoch in range(0,15):
        epoch = 10
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "test/" +
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 
            "_" + self.name + "_" + str(self.BATCH_SIZE) +
            "_epoch_" + str(epoch))
        self.model_path =  self.checkpoint_dir + '/' + self.name + '/epoch_' + str(epoch)
        self.generator = tf.keras.models.load_model(self.model_path, custom_objects={'tf': tf})

        # Check its architecture
        # self.generator.summary()
        # for step, (input_image, target, video_name, input_image_raw) in tqdm(self.test_dataset.enumerate()):
        #     actual_video = np.array(video_name)
        #     # print(actual_video)
            
        #     if step == 0:
        #         previous_video = actual_video
                
        #     if actual_video != previous_video:
        #         previous_video = actual_video
        #         self.generate_video(self.images_video_list, np.array(step, dtype=np.int64), epoch)
        #         self.images_video_list = []
        #         # break
                
        #     self.test_step(input_image, target, step, input_image_raw, dataset = 'bdda')
        # ---------------------------------------------------------------------------------
        # self.mean_kld_bdda = np.mean(self.list_kld)
        # self.mean_cc_bdda = np.mean(self.list_cc)
        # self.mean_sAUC_bdda = np.mean(self.list_sAUC)


        # self.list_kld = []
        # self.list_cc = []
        # self.list_sAUC = []
        # self.best_kld = 1
        # self.best_cc = 0
        for step, (input_image, target, video_name, input_image_raw) in tqdm(self.test_dataset_DADA.enumerate()):
            actual_video = np.array(video_name)
            # print(actual_video)
            if step == 0:
                previous_video = actual_video
                
            if actual_video != previous_video:
                previous_video = actual_video
                self.generate_video_dada(self.images_video_list, np.array(step, dtype=np.int64), epoch)
                self.images_video_list = []
            self.test_step(input_image, target, step, input_image_raw, dataset = 'dada')
            

        # self.mean_kld_dada = np.mean(self.list_kld)
        # self.mean_cc_dada = np.mean(self.list_cc)
        # self.mean_sAUC_dada = np.mean(self.list_sAUC)

        # header = ['model',
        #             'epoch',
        #             'Batch size', 
        #             'KLD', 
        #             'CC', 
        #             's-AUC', 
        #             'KLD', 
        #             'CC', 
        #             's-AUC']
        # data = [self.name, epoch, 
        #         self.BATCH_SIZE,  
        #         self.mean_kld_bdda, 
        #         self.mean_cc_bdda,
        #         self.mean_sAUC_bdda,
        #         self.mean_kld_dada, 
        #         self.mean_cc_dada, 
        #         self.mean_sAUC_dada]

        # with open('results.csv', 'a', encoding='UTF8', newline='') as f:
        #     writer = csv.writer(f)

        #     # write the header
        #     writer.writerow(header)

        #     writer.writerow(data)

if __name__ == "__main__":
    test = TestModel()
    test.main()  

