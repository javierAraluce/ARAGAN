import tensorflow as tf
from typing import Tuple

class Dataloader(object):
    def __init__(self, width: int, height: int, output_channels: int) ->None:
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height  
        self.OUTPUT_CHANNELS = output_channels

    def load(self, image_file: str) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Load dataset function
        
        Args:
            image_file (str): Image file name

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: input image and attention image from dataset
        '''
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        input_image = tf.image.decode_jpeg(image, channels=3)

        # Get attention map file name replacing the text in the file name
        map_file = tf.strings.regex_replace(image_file,
                                            "camera_images",
                                            "gazemap_images_resized")
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(map_file)
        map_image = tf.image.decode_jpeg(image, channels=self.OUTPUT_CHANNELS)

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        map_image = tf.cast(map_image, tf.float32)

        return input_image, map_image

    def resize(self, 
               input_image: tf.Tensor, 
               map_image: tf.Tensor,
               height: int, 
               width: int) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Resize both images

        Args:
            input_image (tf.Tensor): RGB image
            map_image (tf.Tensor): Attention image
            height (int): height image size
            width (int): width image size

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Resized RGB image and attention image 
        '''
        input_image = tf.image.resize(
            input_image, 
            [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        map_image = tf.image.resize(
            map_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, map_image

    def random_crop(self,
                    input_image: tf.Tensor, 
                    map_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Random jittering and mirroring to preprocess the training set

        Args:
            input_image (tf.Tensor): RGB image
            map_image (tf.Tensor): Attention image
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Ramdomly cropped RGB image and attention image 
        '''
        stacked_image = tf.stack([input_image, map_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        return cropped_image[0], cropped_image[1]

    def normalize(self, 
                  input_image: tf.Tensor,
                  map_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Normalize the image between 0 and 1

        Args:
            input_image (tf.Tensor): RGB image
            map_image (tf.Tensor): Attention image

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Normalize RGB image and attention image 
        '''
        # Normalizing the images to [0, 1]
        input_image = (input_image / 255.0)
        map_image = (map_image / 255.0)

        return input_image, map_image

    @tf.function()
    def random_jitter(self, 
                      input_image: tf.Tensor, 
                      map_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Random jittering and mirroring to preprocess the training set

        Args:
            input_image (tf.Tensor): RGB image
            map_image (tf.Tensor): Attention image

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Random jittering and mirroring RGB image and attention image 
        '''
        # Resizing to 286x286
        input_image, map_image = self.resize(input_image, map_image, 286, 286)

        # Random cropping back to 256x256
        input_image, map_image = self.random_crop(input_image, map_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            map_image = tf.image.flip_left_right(map_image)

        return input_image, map_image

    def load_image_train(self, image_file: str) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Training dataloader pipeline

        Args:
            image_file (str): Image file name

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: RGB image and attention image 
        '''
        input_image, map_image = self.load(image_file)
        input_image, map_image = self.random_jitter(input_image, map_image)
        # input_image, map_image = self.resize(input_image, map_image,
        #                                 self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, map_image = self.normalize(input_image, map_image)
        return input_image, map_image

    def load_image_test(self, image_file: str) -> Tuple[tf.Tensor, tf.Tensor]:
        '''Testing dataloader pipeline

        Args:
            image_file (str): Image file name

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: RGB image and attention image 
        '''
        input_image, map_image = self.load(image_file)
        input_image, map_image = self.resize(input_image, map_image,
                                        self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, map_image = self.normalize(input_image, map_image)

        return input_image, map_image

