import tensorflow as tf

class Modules(object):
    def __init__(self):
        print('Blocks init')

    def down_res_block(self, 
                       input: tf.Tensor, 
                       filters: int,
                       stride = 2) -> tf.Tensor:
        '''Downsample residual block 

        Args:
            input (tf.Tensor): input tensor
            filters ((int, int)): Tuple with filters for the convolutions
            stride (int, optional): stride paramters for dimension reduction.
            Defaults to 2.

        Returns:
            tf.Tensor: output tensor
        '''
        # Store the input the tensor for the residual connection 
        x_skip = input
        # Set the filters for the different convolutions of the block
        f1, f2 = filters
        # Dropout parameter 
        dropout = 0.5

        # First block
        x = tf.keras.layers.Conv2D(
            f1, 
            kernel_size=(1, 1), 
            strides=(stride, stride), 
            padding='valid', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001))(input)
        # When s = 2 then it is like downsizing the feature map
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        # Second block
        x = tf.keras.layers.Conv2D(
            f1, 
            kernel_size=(3, 3),
            strides=(1, 1), 
            padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        #third block
        x = tf.keras.layers.Conv2D(
            f2, 
            kernel_size=(1, 1), 
            strides=(1, 1), 
            padding='valid', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # shortcut 
        x_skip = tf.keras.layers.Conv2D(
            f2, 
            kernel_size=(1, 1), 
            strides=(stride, stride), 
            padding='valid', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
        x_skip = tf.keras.layers.BatchNormalization()(x_skip)

        # Add residual connection
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        
        return x

    def upsample(self, input: tf.Tensor, filters: int) -> tf.Tensor:
        '''Upsample block

        Args:
            input (tf.Tensor): Input tensor
            filters (_type_): filters for the Convolution Transpose Kernel

        Returns:
            tf.Tensor: Output tensor
        '''
        # Random normal kernel initilaizer 
        initializer = tf.random_normal_initializer(mean = 0., stddev = 0.02)
        # Transpose Kernel with stride = 2 for dimensional increasing 
        x = tf.keras.layers.Conv2DTranspose(filters,
                                            kernel_size = 3,
                                            strides=2,
                                            padding='same', 
                                            kernel_initializer=initializer,
                                            use_bias=False)(input)

        # BatchNormalization and Relu activation
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        return x

    def upsample_unet(self, 
                      filters: int, 
                      size: int, 
                      apply_batchnorm: bool = True, 
                      apply_dropout: bool = False, 
                      dropout: float = 0.5,
                      activator: str = 'relu') -> tf.keras.Sequential:
        '''Upsample block for the Unet 

        Args:
            filters (int): Filters for the Convolution Transpose Kernel
            size (int): kernel size for the Convolution Transpose Kernel
            apply_batchnorm (bool, optional): Set if BatchNormalization is 
            applied. Defaults to True.
            apply_dropout (bool, optional): Set if Dropoout is applied. Defaults
            to False.
            dropout (float, optional): Dropout value. Defaults to 0.5.
            activator (str, optional): Activator. Defaults to 'relu'.

        Returns:
            tf.keras.Sequential: Model output with the block 
        '''
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters,
                                            size, 
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(dropout))

        if activator == 'leaky_relu':
            result.add(tf.keras.layers.LeakyReLU())
        elif activator == 'relu':
            result.add(tf.keras.layers.ReLU())

        return result
        
    def downsample(self, 
                   filters: int, 
                   size: int, 
                   apply_batchnorm: bool = True , 
                   apply_dropout: bool = False, 
                   dropout: float = 0.5,
                   activator: str = 'leaky_relu',
                   stride: int = 2, 
                   apply_pooling: bool = False) -> tf.keras.Sequential:
        '''Downsample block 

        Args:
            filters (int): Filters for the Convolution Transpose Kernel
            size (int): kernel size for the Convolution Transpose Kernel
            apply_batchnorm (bool, optional): Set if BatchNormalization is 
            applied. Defaults to True.
            apply_dropout (bool, optional): Set if Dropoout is applied. Defaults
            to False.
            dropout (float, optional): Dropout value. Defaults to 0.5.
            activator (str, optional): Activator. Defaults to 'leaky_relu'
            stride (int, optional): stride paramters for dimension reduction.
            Defaults to 2.
            apply_pooling (bool, optional): Set if Pooling is applied. Defaults
            to False.

        Returns:
            tf.keras.Sequential: Model output with the block 
        '''
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, 
                                   size,
                                   strides = stride, 
                                   padding='same',
                                   kernel_initializer=initializer, 
                                   use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        if apply_pooling:
            result.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                                                    strides=None,
                                                    padding="valid", 
                                                    data_format=None))  

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(dropout))

        if activator == 'leaky_relu':
            result.add(tf.keras.layers.LeakyReLU())
        elif activator == 'relu':
            result.add(tf.keras.layers.ReLU())

        return result

    def google_attention(self, 
                         inputs: tf.Tensor, 
                         filters: int,
                         ratio: int, 
                         kernel_size: int) -> tf.Tensor: 
        '''Self Attention block 

        Args:
            inputs (tf.Tensor): Input tensor 
            filters (int): filters for the attention module that are going to be 
            divided by the ratio to avoid bottleneck
            ratio (int): ratio to avoid bottleneck
            kernel_size (int): kernel size

        Returns:
            tf.Tensor: Output Tensor
        '''
        # Store input shape
        input_shape = inputs.shape
        # print(batch_size, h, w, num_channels)

        # Residual connection
        x_skip = inputs
        # n = h * w
        bottleneck = int(filters / ratio)
        bottleneck_2 = int(filters / 2)

        # Query [bs,h,w,c']
        q = tf.keras.layers.Conv2D(filters = bottleneck,  
                                   kernel_size = 1, 
                                   strides = 1, 
                                   padding = 'same')(inputs) 
        q = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=2, 
                                         padding='same')(q)

        # Keys [bs,h,w,c']
        k = tf.keras.layers.Conv2D(filters = bottleneck, 
                                   kernel_size = 1, 
                                   strides = 1, 
                                   padding = 'same')(inputs) 

        # Values [bs,h,w,c]
        v = tf.keras.layers.Conv2D(filters = bottleneck_2, 
                                   kernel_size = 1, 
                                   strides = 1, 
                                   padding = 'same')(inputs) 
        v = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=2, 
                                         padding='same')(v)

        # Reshaping operations to matrix calculation 

        q = tf.reshape(q,(-1, tf.multiply(q.shape[1],q.shape[2]), q.shape[3])) 
        k = tf.reshape(k,(-1, tf.multiply(k.shape[1],k.shape[2]), k.shape[3])) 
        v = tf.reshape(v,(-1, tf.multiply(v.shape[1],v.shape[2]), v.shape[3])) 

        # N = h * w
        s = tf.matmul(k, q, transpose_b=True) # [bs, N, N]

        s = tf.nn.softmax(s) # scores

        gamma = tf.Variable(0.0, trainable=True , name = "gamma")
        o = tf.matmul(s, v) # [bs, N, C]
        shape_o = o.shape
        
        o = tf.reshape(o,
                       (input_shape[0], 
                        input_shape[1], 
                        input_shape[2],
                        shape_o[2])) # [bs, h, w, C]

        conv = tf.keras.layers.Conv2D(filters = input_shape[3], 
                                      kernel_size = 1, 
                                      strides = 1, 
                                      padding = 'same')(o)

        output = tf.keras.layers.Add()([gamma * conv, x_skip])

        return output   

    def ResBlock_CBAM(self, 
                      inputs: tf.Tensor, 
                      filters: int, 
                      ratio: int, 
                      kernel_size: int,
                      stride: int) -> tf.Tensor:
        '''Convolutional Block Attention module

        Args:
            inputs (tf.Tensor): Input tensor
            filters (int): convolution filters
            ratio (int): ratio to avoid bottleneck
            kernel_size (int): convolutional kernel size
            stride (int): stride paramters for dimension reduction

        Returns:
            tf.Tensor: _description_
        '''
       
        # x_skip = inputs
        dropout = 0.5
        regularizer = tf.keras.regularizers.l2(0.001)                         

        x = tf.keras.layers.Conv2D(filters = filters, 
                                   kernel_size=(1, 1), 
                                   strides=stride, 
                                   padding='valid', 
                                   kernel_regularizer=regularizer)(inputs)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        
        x_skip = x

        # ChannelAttention
        
        # Average of every layer
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)

        shared_layer_one = tf.keras.layers.Dense(filters//ratio,
                                                 activation='relu', 
                                                 kernel_initializer='he_normal', 
                                                 use_bias=True, 
                                                 bias_initializer='zeros')

        shared_layer_two = tf.keras.layers.Dense(filters,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,           
                                                 bias_initializer='zeros') 

        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        # MaxPool
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(x) # Max of every layer
        # max_pool = tf.keras.layers.Reshape((1,1,filters))(max_pool)

        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)


        attention = tf.keras.layers.Add()([avg_pool,max_pool])
        attention = tf.keras.layers.Activation('sigmoid')(attention)
        
        x_channel = tf.keras.layers.Multiply()([x, attention])

        x = tf.keras.layers.multiply(inputs = [x, x_channel])

        # SpatialAttention
        conv2d = tf.keras.layers.Conv2D(filters = 1,
                                        kernel_size=kernel_size,
                                        strides=1,
                                        padding='same',
                                        activation='sigmoid',
                                        kernel_initializer='he_normal',
                                        use_bias=False)
        avg_pool = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(x)
                    
        # MaxPool
        max_pool = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(x)

        attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

        attention = conv2d(attention)
        
        x = tf.keras.layers.multiply(inputs = [x, attention])

        # x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.multiply(inputs = [x, x_skip])

        return x 

    def MultiHead_attention_block(self,
                                  input: tf.Tensor, 
                                  filters: int ) -> tf.Tensor:
        '''Multi Head Attention Block, heads are built with self attention

        Args:
            input (tf.Tensor): Input tensor
            filters (int): filters of the self-attention module

        Returns:
            tf.Tensor: Output tensor
        '''
        heads_length = 8
        x_head = input
        x = input
        for head in range(1, heads_length):
            multi_head =  self.google_attention(inputs = x_head,
                                                filters = filters,
                                                ratio = 8,
                                                kernel_size = 3)
            x = tf.keras.layers.concatenate([x, multi_head])


        x = tf.keras.layers.Conv2D(filters = filters, 
                                   kernel_size = 1, 
                                   strides = 1, 
                                   padding = 'same')(x) 

        # Layer Normalization
        x = tf.keras.layers.Add()([x, x_head]) 
        x = tf.keras.layers.LayerNormalization()(x) 

        return x