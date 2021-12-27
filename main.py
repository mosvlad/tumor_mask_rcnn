import os
import re
import sys
import cv2
import glob
import time
import json
import math
import random
import imutils
import itertools
import matplotlib
import numpy as np
import skimage.draw
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.cm as cm
from skimage.io import imread
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
#from imgaug import augmenters as iaa

sys.path.append("./Mask_RCNN")

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log

if __name__ == "__main__":


    # Directory to save logs and trained model
    MODEL_DIR = os.path.join("./Mask_RCNN", "logs")
    DEFAULT_LOGS_DIR = 'logs'

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join("./Mask_RCNN", "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Import COCO config
    sys.path.append(os.path.join("./Mask_RCNN", 'samples/coco/'))
    #import coco

    plt.rcParams['figure.facecolor'] = 'white'

    DATASET_DIR = './brain-tumor-segmentation/brain_tumor_data/'

    examples = [Image.open(DATASET_DIR + 'train/131.jpg'), Image.open(DATASET_DIR + 'train/116.jpg'), Image.open(DATASET_DIR + 'train/221.jpg')]
    examplesSeg = ['131.jpg28778', '116.jpg10596', '221.jpg19584']

    fig = plt.figure(figsize=(10, 10))

    for i in range(0, len(examples)):
        a = fig.add_subplot(2, 3, i+1)
        imgplot = plt.imshow(examples[i])
        a.set_title('Example '+str(i))

    with open(DATASET_DIR+'train/annotations.json') as json_file:
        data = json.load(json_file)
        for i in range(0,len(examplesSeg)):
            coord = list(zip(data[examplesSeg[i]]['regions'][0]['shape_attributes']['all_points_x'],data[examplesSeg[i]]['regions'][0]['shape_attributes']['all_points_y']))
            image = Image.new("RGB", np.asarray(examples[i]).shape[0:2])
            draw = ImageDraw.Draw(image)
            draw.polygon((coord), fill=200)
            a = fig.add_subplot(2, 3, 3+i+1)
            imgplot = plt.imshow(image)
            a.set_title('Segment for example ' + str(i))
        plt.show()

