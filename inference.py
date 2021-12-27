import os
import matplotlib.pyplot as plt

import train

ROOT_DIR = os.path.abspath("./Mask_RCNN")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import visualize

from train import TrainConfig
from train import TumorDataset

def display_image(dataset, ind):
    plt.figure(figsize=(5,5))
    plt.imshow(dataset.load_image(ind))
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')
    #plt.show()


def get_ax(rows=1, cols=1, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def predict_and_plot_differences(dataset, img_id):
    config = train.TrainConfig()
    original_image, image_meta, gt_class_id, gt_box, gt_mask =\
        modellib.load_image_gt(dataset, config,  img_id, use_mini_mask=False)

    results = model.detect([original_image], verbose=1)
    r = results[0]
    print(r)


    visualize.display_differences(
        original_image,
        gt_box, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'],
        class_names = ['tumor'], title="", ax=get_ax(),
        show_mask=True, show_box=True)


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights // specify path or load last
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


DATASET_DIR = './brain-tumor-segmentation/brain_tumor_data/'
dataset_val = TumorDataset()
dataset_val.load_brain_tumor_images(DATASET_DIR, 'val')
dataset_val.prepare()

DATA_ID = 19

display_image(dataset_val, DATA_ID)
predict_and_plot_differences(dataset_val, DATA_ID)
plt.show()