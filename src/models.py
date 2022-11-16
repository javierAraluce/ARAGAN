import tensorflow as tf
from layers import Custom_Layers
from typing import Tuple

class Models(object):
    def __init__(self,
                 img_height: int, 
                 img_width: int, 
                 output_channels: int, 
                 batch_size: int):
        self.blocks = Custom_Layers()
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.OUTPUT_CHANNELS = output_channels
        self.batch_size = batch_size

    def Unet(self):
        inputs = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 3], 
            batch_size = self.batch_size)
        kernel_size = 5 

        down_stack = [
            self.blocks.downsample( 64, 
                                   kernel_size,
                                   apply_batchnorm = False, 
                                   apply_dropout = True, 
                                   dropout = 0.5,
                                   activator = 'relu'),  # (bs, 128, 128, 64)
            self.blocks.downsample(128, 
                                   kernel_size,
                                   apply_batchnorm = False, 
                                   apply_dropout = True,
                                   dropout = 0.5, 
                                   activator = 'relu'),  # (bs, 64, 64, 128)
            self.blocks.downsample(256, 
                                   kernel_size, 
                                   apply_batchnorm = False,
                                   apply_dropout = True, 
                                   dropout = 0.5,
                                   activator = 'relu'),  # (bs, 32, 32, 256)
            self.blocks.downsample(512,
                                   kernel_size,
                                   apply_batchnorm = False,
                                   apply_dropout = False, 
                                   dropout = 0.5,
                                   activator = 'relu'),  # (bs, 16, 16, 512)
            self.blocks.downsample(512, 
                                   kernel_size, 
                                   apply_batchnorm = False,
                                   apply_dropout = False,
                                   dropout = 0.5, 
                                   activator = 'relu'),  # (bs, 8, 8, 512)
            self.blocks.downsample(512,
                                   kernel_size,
                                   apply_batchnorm = False,
                                   apply_dropout = False, 
                                   dropout = 0.5,
                                   activator = 'relu'),  # (bs, 4, 4, 512)
            self.blocks.downsample(512,
                                   kernel_size,
                                   apply_batchnorm = False,
                                   apply_dropout = False,
                                   dropout = 0.5,
                                   activator = 'relu'),  # (bs, 2, 2, 512)
            self.blocks.downsample(512, 
                                   kernel_size, 
                                   apply_batchnorm = False,
                                   apply_dropout = False,
                                   dropout = 0.5,
                                   activator = 'relu'),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.blocks.upsample_unet(512, 
                                      kernel_size, 
                                      apply_batchnorm = True, 
                                      apply_dropout = True, 
                                      dropout = 0.5, 
                                      activator = 'relu'),  # (bs, 1, 1, 512)
            self.blocks.upsample_unet(512,
                                      kernel_size,
                                      apply_batchnorm = True, 
                                      apply_dropout = True, 
                                      dropout = 0.5,
                                      activator = 'relu'),  # (bs, 2, 2, 512)
            self.blocks.upsample_unet(512,
                                      kernel_size,
                                      apply_batchnorm = True,
                                      apply_dropout = True, 
                                      dropout = 0.5,
                                      activator = 'relu'),  # (bs, 4, 4, 512)
            self.blocks.upsample_unet(512,
                                      kernel_size, 
                                      apply_batchnorm = True,
                                      apply_dropout = True, 
                                      dropout = 0.5,
                                      activator = 'relu'),  # (bs, 8, 8, 512)
            self.blocks.upsample_unet(512, 
                                      kernel_size, 
                                      apply_batchnorm = True, 
                                      apply_dropout = False,
                                      dropout = 0.5, 
                                      activator = 'relu'),  # (bs, 16, 16, 512)
            self.blocks.upsample_unet(256,
                                      kernel_size,
                                      apply_batchnorm = True,
                                      apply_dropout = False, 
                                      dropout = 0.5,
                                      activator = 'relu'),  # (bs, 32, 32, 256)
            self.blocks.upsample_unet(128,
                                      kernel_size, 
                                      apply_batchnorm = True, 
                                      apply_dropout = False, 
                                      dropout = 0.5, 
                                      activator = 'relu'),  # (bs, 64, 64, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS,
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='sigmoid')  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        
        generator_model = tf.keras.Model(inputs=inputs, outputs=x)
        generator_model.summary()
        return generator_model

    def Resnet(self):
        inputs = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 3], 
            batch_size = self.batch_size)
        kernel_size = 3
        # x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(inputs)
        x = inputs

        stride_steps = 14
        upsample_steps = 0
        # Downsample
        filters = 8
        for deep in range(1, stride_steps):    
            if deep % 2 == 0:
                stride = 2
                upsample_steps += 1
            else:
                stride = 1
            
            filters = filters * stride 
            x = self.blocks.down_res_block(x, 
                                           stride=stride, 
                                           filters=(filters, filters * 4))
        
        # Upsample
        filters = 128
        for deep in range (1,upsample_steps):
            x = self.blocks.upsample(input = x,
                                filters = filters)
            filters = filters / 2

        initializer = tf.random_normal_initializer(0., 0.02)
        x  = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS, 
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='sigmoid')(x)  # (bs, 256, 256, 3)
        
        generator_model = tf.keras.Model(inputs=inputs, outputs=x)
        generator_model.summary()
        return generator_model

    def Resnet_Attention(self):
        inputs = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 3], 
            batch_size = self.batch_size)
        kernel_size = 3
        # x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(inputs)
        x = inputs

        stride_steps = 14
        upsample_steps = 0
        # Downsample
        filters = 8
        for deep in range(1, stride_steps):    
            if deep % 2 == 0:
                stride = 2
                upsample_steps += 1
            else:
                stride = 1
            
            filters = filters * stride 
            x = self.blocks.down_res_block(x, 
                                           stride=stride, 
                                           filters=(filters, filters * 4))

        x =  self.blocks.google_attention(inputs = x,
                                          filters = filters,
                                          ratio = 8,
                                          kernel_size = 3) 
        
        
        # Upsample
        filters = 128
        for deep in range (1, upsample_steps):
            x = self.blocks.upsample(input = x,
                                     filters = filters)
            filters = filters / 2

        initializer = tf.random_normal_initializer(0., 0.02)
        x  = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS, 
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='sigmoid')(x)  # (bs, 256, 256, 3)
        
        generator_model = tf.keras.Model(inputs=inputs, outputs=x)
        generator_model.summary()
        return generator_model

    def Resnet_Multi_Head_Attention(self):
        inputs = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 3], 
            batch_size = self.batch_size)
        kernel_size = 3
        x = inputs

        stride_steps = 8
        upsample_steps = 0
        # Downsample
        filters = 4
        for deep in range(1, stride_steps):    
            if deep % 2 == 0:
                stride = 2
                upsample_steps += 1
            else:
                stride = 1
                
            
            filters = filters * stride 
            x = self.blocks.down_res_block(x,
                                           stride=stride, 
                                           filters=(filters, filters))

        for deep in range(1, stride_steps):    
            if deep % 2 == 0:
                stride = 2
                upsample_steps += 1
            else:
                stride = 1
                
            
            filters = filters * stride 
            x = self.blocks.down_res_block(x,
                                           stride=stride,
                                           filters=(filters, filters))
            x = self.blocks.MultiHead_attention_block(x, filters)
     
        # Upsample
        filters = 128
        for deep in range (1,upsample_steps):
            x = self.blocks.upsample(input = x,
                                     filters = filters)
            filters = filters / 2

        initializer = tf.random_normal_initializer(0., 0.02)
        x  = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS, 
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='sigmoid')(x)  # (bs, 256, 256, 3)
        
        generator_model = tf.keras.Model(inputs=inputs, outputs=x)
        generator_model.summary()
        return generator_model

    def CBAM(self):
        inputs = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 3], 
            batch_size = self.batch_size)
        kernel_size = 3
        x = inputs

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = tf.keras.layers.Conv2D(256, kernel_size=(7, 7), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # Downsample
        filters = 32
        for deep in range(1, 16):    
            if deep % 2 == 0:
                stride = 2
            else:
                stride = 1
            filters = filters * stride  
            x = self.blocks.ResBlock_CBAM(inputs = x, 
                                          filters = filters, 
                                          ratio = 8, 
                                          kernel_size = 7, 
                                          stride = stride)

        # Upsample
        filters = 1024
        for deep in range (1,8):
            x = self.blocks.upsample(input = x,
                                     filters = filters)
            filters = filters / 2

        initializer = tf.random_normal_initializer(0., 0.02)
        x  = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS, 
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='sigmoid')(x)  # (bs, 256, 256, 3)
        
        generator_model = tf.keras.Model(inputs=inputs, outputs=x)
        generator_model.summary()
        return generator_model

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        kernel_size = 3 

        inp = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 3],
            name='input_image', 
            batch_size=self.batch_size)
        tar = tf.keras.layers.Input(
            shape=[self.IMG_HEIGHT, self.IMG_WIDTH, self.OUTPUT_CHANNELS], 
            name='target_image', 
            batch_size=self.batch_size)

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, chan*2)


        down1 = self.blocks.downsample(
            filters = 64,
            size = kernel_size,
            apply_batchnorm = False, 
            apply_dropout = False,
            dropout = 0.5,
            activator = 'leaky_relu')(x)  # (bs, 128, 128, 64)
        down2 = self.blocks.downsample(
            filters = 128,
            size = kernel_size, 
            apply_batchnorm = True,
            apply_dropout = False, 
            dropout = 0.5,
            activator = 'leaky_relu')(down1)  # (bs, 64, 64, 128)
        down3 = self.blocks.downsample(
            filters = 256, 
            size = kernel_size,
            apply_batchnorm = True,
            apply_dropout = False,
            dropout = 0.5,
            activator = 'leaky_relu')(down2)  # (bs, 32, 32, 256)

        # Add zerros to filter borders 
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)

        down4 = self.blocks.downsample(
            filters = 512, 
            size = kernel_size,
            apply_batchnorm = True,
            apply_dropout = False,
            dropout = 0.5,
            activator = 'leaky_relu',
            stride = 1)(zero_pad1)  
        
        # attention = self.blocks.google_attention(inputs = down4,
        #                                          filters = 512,
        #                                          ratio = 8,
        #                                          kernel_size = 3)
         
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(down4) # (bs,33,33, 512)


        last = tf.keras.layers.Conv2D(
            filters = 1,
            kernel_size = 4, 
            strides=1,
            kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        discriminator_model = tf.keras.Model(inputs=[inp, tar], outputs=last)
        # discriminator_model.summary()
    
        return discriminator_model
