# ARAGAN: A dRiver Attention estimation model based on conditional Generative Adversarial Network

## **Published at [IV 2022](enlace)!**

## Authors
				
[Javier Araluce](javier.araluce@uah.es), [Luis Miguel Bergasa](luism.bergasa@uah.es), [Manuel Ocaña](mocana@depeca.uah.es), [Rafael Barea](rafael.barea@uah.es), [Elena López-Guillén](elena.lopezg@uah.es)  and [Pedro Revenga](pedro.revenga@uah.es)

## Overview

![framework](https://github.com/javierAraluce/ARAGAN/blob/main/figures/ARAGAN-Framework.drawio.png)

Predicting driver’s attention in complex driving scenarios is becoming a hot topic due to it helps the design of some autonomous driving tasks, optimizing visual scene understanding and contributing knowledge to the decision making.
We introduce ARAGAN, a driver attention estimation model based on a conditional Generative Adversarial Network (cGAN). 
This architecture uses some of the most challenging and novel deep learning techniques to develop this task.
It fuses adversarial learning with Multi-Head Attention mechanisms. To the best of our knowledge, this combination has never been applied to predict driver’s attention.
Adversarial mechanism learns to map an attention image from an RGB traffic image while mapping the loss function. Attention mechanism contributes to the deep learning paradigm finding the most interesting feature maps inside the tensors of the net. In this work, we have adapted this concept to find the saliency areas in a driving scene.
 
An ablation study with different architectures has been carried out, obtained the results in terms of some saliency metrics. Besides, a comparison with other state-of-the-art models has been driven, outperforming results in accuracy and performance, and showing that our proposal is adequate to be used on real-time applications. 
ARAGAN has been trained in BDDA and tested in BDDA and DADA2000, which are two of the most complex driver attention datasets available for research. 			
				
				
## Requirements
 
This Repository has been tested using `Tensorflow=2.4` and `CUDA=11.0`, `CUDNN=8`
```bash
pip3 install -r requiriments.txt
```


### BDDA Dataset
Images have been sampled at 10 Hz from the videos obtained from https://bdd-data.berkeley.edu/. Click on the "Download Dataset" to get to the user portal and then you will find the BDD-Attention dataset
#### Training set parser
- Parse RGB videos
```bash
python3 src/data/parse_videos.py --video_dir dataset/BDDA/training/camera_videos --image_dir dataset/BDDA/training/camera_images/all_images
```
- Parse Attention maps videos
```bash
python3 src/data/parse_videos.py --video_dir dataset/BDDA/training/gazemap_videos --image_dir dataset/BDDA/training/camera_images/gazemap_images
```
- Resized attention map images
```bash
python3 src/data/gaze_map_image_normalization.py --image_dir dataset/BDDA/training/gazemap_videos --image_dir_resized dataset/BDDA/training/camera_images/gazemap_images_resized
```


#### Validation set parser
- Parse RGB videos
```bash
python3 src/data/parse_videos.py --video_dir dataset/BDDA/validation/camera_videos --image_dir dataset/BDDA/validation/camera_images/all_images
```
- Parse Attention maps videos
```bash
python3 src/data/parse_videos.py --video_dir dataset/BDDA/validation/gazemap_videos --image_dir dataset/BDDA/validation/camera_images/gazemap_images
```
- Resized attention map images
```bash
python3 src/data/gaze_map_image_normalization.py --image_dir dataset/BDDA/validation/gazemap_videos --image_dir_resized dataset/BDDA/validation/camera_images/gazemap_images_resized
```

#### Testing set parser

- Parse RGB videos
```bash
python3 src/data/parse_videos.py --video_dir dataset/BDDA/tes/camera_videos --image_dir dataset/BDDA/tes/camera_images/all_images
```
- Parse Attention maps videos
```bash
python3 src/data/parse_videos.py --video_dir dataset/BDDA/tes/gazemap_videos --image_dir dataset/BDDA/tes/camera_images/gazemap_images
```
- Resized attention map images
```bash
python3 src/data/gaze_map_image_normalization.py --image_dir dataset/BDDA/tes/gazemap_videos --image_dir_resized dataset/BDDA/tes/camera_images/gazemap_images_resized
```

Clean data, to have the same frames in RGB videos than in attention maps
```bash
python3 clean_data.py
```

### DADA-2000 Dataset
DADA2000 dataset (about 53GB with compresed mode) can be downloaded from [here](https://pan.baidu.com/s/1RfNjeW0Rjj6R4N7beSTYrA). (Extraction code: 9pab) 

#### Training set parser
- Parse RGB and attention map videos from training set
```bash
python3 src/data/parse_DADA_2000_dataset.py --dataset_set training
```
#### Validation set parser
- Parse RGB and attention map videos from validation set
```bash
python3 src/data/parse_DADA_2000_dataset.py --dataset_set validation
```

#### Testing set parser
- Parse RGB and attention map videos from testing set
```bash
python3 src/data/parse_DADA_2000_dataset.py --dataset_set testing
```

### Dataset structure 
Dataset structure have to be in the following way to work with the code:
* dataset
  * BDDA
      * test
        * camera_images
        * gazemap_images_resized

      * training
        * camera_images
        * gazemap_images_resized

      * validation
        * camera_images
        * gazemap_images_resized

    * DADA2000
      * test
        * camera_images
        * gazemap_images_resized
      * training
        * camera_images
        * gazemap_images_resized
      * validation
        * camera_images
        * gazemap_images_resized


				

## Parameters.
				
				
				
				
## Description

To train the model you will have to launch:

```bash
python3 src/train.py
```
The script will ask you to set the Generator that you want to Train from a list (`['CBAM', 'Resnet', 'Resnet_Attention', 'Resnet_Multi_Head_Attention', 'Unet']`), these Generators will be built with these blocks as the downsample section, to further analysis go to the paper:

### Residual convolotutional module (Resblock)
![resBlock](https://github.com/javierAraluce/ARAGAN/blob/main/figures/ARAGAN-Down_resBlock.drawio.png)

### Convolutional Block Attention Module (CBAM)

![cbam](https://github.com/javierAraluce/ARAGAN/blob/main/figures/ARAGAN-Cbam.drawio.png)

### Self-Attention module
![sa](https://github.com/javierAraluce/ARAGAN/blob/main/figures/ARAGAN-google_attention.drawio.png)

### Multi-Head Attention module
![mha](https://github.com/javierAraluce/ARAGAN/blob/main/figures/ARAGAN-Multi-Head.drawio.png)
				
## Results

Here you can see some results of the model in different images from the BDDA testing set

![bdda](https://github.com/javierAraluce/ARAGAN/blob/main/figures/images_output_bdda.png)

And here other result from 
![dada](https://github.com/javierAraluce/ARAGAN/blob/main/figures/images_output_dada.png)
 
				
				
## Future Works

Train in DADA2000 and see the difference with the previous model obtained trained only in BDDA
Test these architecture in other image to image applications like semantic segmentaion, depth estimation. 


# Citing
If you used our work, please cite our work:
```bash
@inproceedings{araluce2022aragan,  
  title = {ARAGAN: A dRiver Attention estimation model based on conditional Generative Adversarial Network},  
  author = {Araluce, Javier and Bergasa, Luis Miguel and Ocaña, Manuel, and Barea, Rafael and L{\'o}pez-Guill{\'e}n, Elena and Revenga, Pedro},  
  booktitle = {2022 IEEE Intelligent Vehicles Symposium (IV)}  
  year = {2022},  
  organization = {IEEE}
}
```

## TODO
- [] Build a DockerFile for this repository
- [] Upload a docker image to DokerHub 


Models available: ['22M_No_attention_8', '24M_ratio_2_8', '95M', 'Attention_skip_3M_8', 'Attention_skip_8', 'Attention_skip_correct_8', 'Attention_skip_per_layer_8', 'CBAM_8', 'CBAM_88M_8', 'CBAM_simple_disc_8', 'MultiHead_80M_8', 'MultiHead_conv_output_22M_8', 'MultiHead_with_LN_and_residual_13M_8', 'MultiHead_with_output_conv_22M_8', 'Multi_Head_25M_8', 'Multi_Head_deeper_8', 'Multi_Head_deeper_86M_8', 'Multi_Head_end_30M_8', 'Multi_attention_8', 'Multi_head_BN_8', 'Multi_head_constant_learning_rate_8', 'Multi_head_middle_8', 'Multi_head_middle_output_size3_8', 'Multi_head_middle_output_size3_random_jitter_8', 'Resblock+MultiHead_16', 'Resblock+MultiHead_deeper_8', 'Resnet_8', 'Resnet_Attention_8', 'Resnet_Multi_Head_Attention_8', 'Resnet_attention_disc_simple_8', 'Resnet_disc_simple_8', 'Unet_8', 'Unet_disc_simple_8', 'test_results_8', 'teste_output_size_8']
Select model?                                 


MultiHead_80M_8
Resnet_Multi_Head_Attention_8
Resblock+MultiHead_deeper_8
Multi_head_BN_8
MultiHead_with_LN_and_residual_13M_8     mejor
MultiHead_conv_output_22M_8   epoch = 10 