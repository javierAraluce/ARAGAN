from tqdm import tqdm
import os
import shutil
from PIL import Image
import argparse

def main_rgb(args):
    dataset_set = args.dataset_set
    PATH_dataset_DADA = 'dataset/DADA2000/'
    dada_path = str(PATH_dataset_DADA + dataset_set + '/')
    

    for category in tqdm(range(1, 54)):
        path = dada_path + 'rgb/' + str(category) + '/'

        events = os.listdir(path)
        for event in events:
            path_events = path + str(event)
            images = os.listdir(path_events)
            for image in images:
                path_file = path_events + '/' + str(image) 
                path_destination = dada_path 
                path_destination += 'camera_images/'
                path_destination += 'all_images/'
                path_destination += str(category) + '_'
                path_destination += str(event) + '_' 
                path_destination += str(image)
                shutil.copy2(path_file, path_destination)
                # print(path_file)
                # print(path_destination)
        
def main_gaze_map(args):
    dataset_set = args.dataset_set
    PATH_dataset_DADA = 'dataset/DADA2000/'
    dada_path = str(PATH_dataset_DADA + dataset_set + '/')
    

    for category in tqdm(range(1, 54)):
        path = dada_path + 'focus/' + str(category) + '/'

        events = os.listdir(path)
        for event in events:
            path_events = path + str(event)
            images = os.listdir(path_events)
            for image in images:
                path_file = path_events + '/' + str(image)
                path_destination = dada_path 
                path_destination += 'gazemap_images_resized/'
                path_destination += 'all_images/'
                path_destination += str(category) + '_'
                path_destination += str(event) + '_' 
                path_destination += str(image) 
                # shutil.copy2(path_file, path_destination)
                im = Image.open(path_file)
                path_destination_jpg = path_destination.replace("png", "jpg")
                im.save(path_destination_jpg, quality=95)

                # print(path_file)
                # print(path_destination)
       




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_set',
        type=str,
        default='training',
        help='the directory that contains videos to parse')
    args = parser.parse_args()
    main_rgb(args)
    main_gaze_map(args)