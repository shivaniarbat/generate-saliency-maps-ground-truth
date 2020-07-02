import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
from PIL import Image
import numpy as np

def generateSM(img_path):
    # read
    img = cv2.imread(img_path)

    # initialize
    imgsize = img.shape
    img_width = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

    # computation
    saliency_map = sm.SMGetSM(img)
    salient_region = sm.SMGetSalientRegion(img)

    saliency_map *= 255.0/np.array(saliency_map).max()

    return np.array(saliency_map).round()

    # save it to the output folder
    # '/sm/.."
    # '/sm-regions/.."



if __name__=='__main__':
    # list all the file names to generate saliency maps of
    # loop through all the input files to generate saliency maps and save them to output folder
    img_path = 'log-images/log_1698.labels0039.tif'

    print(type(img_path))
    img = cv2.imread(img_path)
    sm_map = generateSM(img_path)
    plt.imshow(sm_map,'gray')
    plt.show()
    cv2.imwrite("output/output1.jpg", np.array(sm_map))
