import numpy as np
import os
from tqdm import tqdm

# path of training images
imgs_train_path = 'dataset/BDDA/training/camera_images/'
# path of training maps
maps_train_path = 'dataset/BDDA/training/gazemap_images_resized/'

# path of validation images
imgs_val_path = 'dataset/BDDA/validation/camera_images/'
# path of validation maps
maps_val_path = 'dataset/BDDA/validation/gazemap_images_resized/'

# path of test images
imgs_test_path = 'dataset/BDDA/test/camera_images/'
# path of test maps
maps_test_path = 'dataset/BDDA/test/gazemap_images_resized/'

def clean_dataset():
    '''
    Clean dataset
    '''
    cont_train = 0
    cont_val = 0
    cont_test = 0
    data_names = [str(f) for f in os.listdir(imgs_train_path + 'all_images/') if f.endswith('.jpg')]
    for image in tqdm(data_names):
        seq, frame = image.split("_")
        if not (os.path.exists(maps_train_path + 'all_images/' + seq +  '_' + frame)):
            cont_train += 1
            # print(maps_train_path + 'all_images/' + seq + '_' + frame)
            os.remove(imgs_train_path + 'all_images/' + seq + "_" + frame)

    data_names = [str(f) for f in os.listdir(maps_train_path + 'all_images/') if f.endswith('.jpg')]
    for image in tqdm(data_names):   
        seq, frame = image.split("_")     
        if not (os.path.exists(imgs_train_path + 'all_images/' + seq + '_' + frame)):
            cont_train += 1
            # print(imgs_train_path + 'all_images/' + seq + '_' + frame)
            os.remove(maps_train_path + 'all_images/' + seq + '_' + frame)    

    data_names = [str(f) for f in os.listdir(imgs_val_path + 'all_images/') if f.endswith('.jpg')]
    for image in tqdm(data_names):
        seq, frame = image.split("_")
        if not (os.path.exists(maps_val_path + 'all_images/'+ seq + "_pure_hm_" + frame)):
            cont_val += 1
            # print(maps_val_path + 'all_images/' + seq + "_pure_hm_" + frame)
            os.remove(imgs_val_path + 'all_images/' + seq + "_" + frame)

    data_names = [str(f) for f in os.listdir(maps_val_path + 'all_images/') if f.endswith('.jpg')]
    for image in tqdm(data_names):   
        seq, trash1, trash2, frame = image.split("_")     
        if not (os.path.exists(imgs_val_path + 'all_images/' + seq + '_' + frame)):
            cont_val += 1
            # print(imgs_val_path + 'all_images/' + seq + '_' + frame)
            os.remove(maps_val_path + 'all_images/' + seq + '_pure_hm_' + frame) 

    data_names = [str(f) for f in os.listdir(imgs_test_path + 'all_images/') if f.endswith('.jpg')]
    for image in tqdm(data_names):
        seq, frame = image.split("_")
        if not (os.path.exists(maps_test_path + 'all_images/'+ seq + "_pure_hm_" + frame)):
            cont_test += 1
            print(maps_val_path + 'all_images/' + seq + "_pure_hm_" + frame)
            os.remove(imgs_test_path + 'all_images/' + seq + "_" + frame)

    data_names = [str(f) for f in os.listdir(maps_test_path + 'all_images/') if f.endswith('.jpg')]
    for image in tqdm(data_names):   
        seq, trash1, trash2, frame = image.split("_")     
        if not (os.path.exists(imgs_test_path + 'all_images/' + seq + '_' + frame)):
            # print(imgs_val_path + 'all_images/' + seq + '_' + frame)
            os.remove(maps_test_path + 'all_images/' + seq + '_pure_hm_' + frame)  

    print('cont_train: ', cont_train)
    print('cont_val: ', cont_val)

    print('cont_test: ', cont_test)

if __name__ == "__main__":
    clean_dataset()