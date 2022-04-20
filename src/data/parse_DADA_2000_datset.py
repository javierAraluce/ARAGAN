from tqdm import tqdm
import os
import shutil
from PIL import Image

def main_rgb():
    PATH_dataset_DADA = 'dataset/DADA2000/'
    dada_path = str(PATH_dataset_DADA + 'testing/')
    

    for category in tqdm(range(1, 54)):
        path = dada_path + 'rgb/' + str(category) + '/'

        events = os.listdir(path)
        for event in events:
            path_events = path + str(event)
            images = os.listdir(path_events)
            for image in images:
                path_file = path_events + '/' + str(image) 
                path_destination = dada_path + 'camera_images/' + 'all_images/' + str(category) + '_' + str(event) + '_' + str(image) 
                shutil.copy2(path_file, path_destination)
                # print(path_file)
                # print(path_destination)
        
def main_gaze_map():
    PATH_dataset_DADA = 'dataset/DADA2000/'
    dada_path = str(PATH_dataset_DADA + 'testing/')
    

    for category in tqdm(range(1, 54)):
        path = dada_path + 'focus/' + str(category) + '/'

        events = os.listdir(path)
        for event in events:
            path_events = path + str(event)
            images = os.listdir(path_events)
            for image in images:
                path_file = path_events + '/' + str(image) 
                path_destination = dada_path + 'gazemap_images_resized/' + 'all_images/' + str(category) + '_' + str(event) + '_' + str(image) 
                # shutil.copy2(path_file, path_destination)
                im = Image.open(path_file)
                path_destination_jpg = path_destination.replace("png", "jpg")
                im.save(path_destination_jpg, quality=95)

                # print(path_file)
                # print(path_destination)
       




if __name__ == "__main__":
    # main_rgb()
    main_gaze_map()