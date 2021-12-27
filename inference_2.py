import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.visualize import display_images
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn.model import log

from train import TrainConfig
from train import TumorDataset


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(os.getcwd())
custom_WEIGHTS_PATH = "Mask_RCNN/logs/tumor_detect20211207T1827/mask_rcnn_tumor_detect_0100.h5"


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=7):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

inference_config = InferenceConfig()

DATASET_DIR = './brain-tumor-segmentation/brain_tumor_data/'
dataset_val = TumorDataset()
dataset_val.load_brain_tumor_images(DATASET_DIR, 'val')
dataset_val.prepare()

with tf.device("/cpu:0"):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=inference_config)

print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

from importlib import reload
reload(visualize)

image_id = 3
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
info = dataset_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset_val.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)
r = results[0]
print(r)

visualize.display_differences(
    image,
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    class_names=['tumor'], title="", ax=get_ax(),
    show_mask=True, show_box=True)
plt.show()