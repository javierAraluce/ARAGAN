import cv2
import os
from tqdm import tqdm
import argparse

'''
gaze map 768x1024 -> crop to 576x1024 -> resize to 720x1280
'''
def nomalized_gaze_map(image_dir, image_dir_resized, image_suffix):
    # make sure output directory is there if not create it 
    if not os.path.isdir(image_dir_resized):
        os.makedirs(image_dir_resized)

    image_names = [f for f in os.listdir(image_dir) if f.endswith(image_suffix)]
    for image_file in tqdm(image_names):

        filename = os.path.join(image_dir, image_file)
        filename_resized = os.path.join(image_dir_resized, image_file)

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        crop_image =  img[96:672, 0:1024] # Remove black bars 
        dim =  (1280, 720)
        resized = cv2.resize(crop_image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(filename_resized, resized) 

def main(args):
    # nomalized_gaze_map
    nomalized_gaze_map(args.image_dir, args.image_dir_resized, 
                    image_suffix=args.image_suffix)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',
        type=str,
        default='dataset/BDDA/test/gazemap_images',
        help='the directory that contains videos to parse')
    parser.add_argument('--image_dir_resized',
        type=str,
        default='dataset/BDDA/test/gazemap_images_resized/all_images',
        help='the directory of parsed frame images')
    parser.add_argument('--image_suffix',
        type=str,
        default='.jpg',
        help='the suffix of images files. E.g., .jpg')
    
    args = parser.parse_args()
    
    main(args)