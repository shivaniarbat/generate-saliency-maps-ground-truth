import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
import numpy as np
from os import listdir

def generateSM(img_dir,img_name,output_dir):
    # image path
    img_path = img_dir + '/' + img_name
    print(img_path)
    # read
    img = cv2.imread(img_path)

    # initialize
    imgsize = img.shape
    img_width = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

    # computation
    saliency_map = sm.SMGetSM(img)
    # salient_region = sm.SMGetSalientRegion(img)

    saliency_map *= 255.0/np.array(saliency_map).max()

    saliency_map = np.array(saliency_map).round()

    # save it to the output folder
    # '/sm/.."
    output_file_name = output_dir + img_name.split('.tif')[0] + "-sm.jpg"
    cv2.imwrite(output_file_name, np.array(saliency_map))
    return saliency_map

    # '/sm-regions/.."

if __name__ == '__main__':

    img_dir = '/Users/shivani/Documents/thesis-work /data/Binary-masks-for-data/log_1698'
    output_dir = '/Users/shivani/Documents/thesis-work /data/Generated-saliency-maps/log_1698/'
    input_images = listdir(img_dir)

    for img_name in input_images:
        if not img_name.startswith('.'):
            sm_map = generateSM(img_dir,img_name,output_dir)





