import sys
import os
import json
import skimage
import cv2
import numpy as np

sys.path.append("./Mask_RCNN")
ROOT_DIR = os.path.abspath("./Mask_RCNN")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib


class TrainConfig(Config):
    """Configuration for training on the brain tumor dataset.
    Derives from the base Config class and overrides values specific
    to the brain tumor dataset.
    """

    NAME = "tumor_detect"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2

    DETECTION_MIN_CONFIDENCE = 0.7

    LEARNING_RATE = 0.01

    STEPS_PER_EPOCH = 10

    VALIDATION_STEPS = 5


class TumorDataset(utils.Dataset):
    """Generates the brain tumors dataset and json annotations.
    """

    def load_brain_tumor_images(self, dataset_dir, folder):
        self.add_class("tumor", 1, "tumor")

        assert folder in ["train", "val", 'test']

        dataset_dir = os.path.join(dataset_dir, folder)

        annotations = json.load(open(os.path.join(dataset_dir, 'annotations.json')))

        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for i in annotations:
            polygons = [r['shape_attributes'] for r in i['regions']]  # get segmented regions from json file

            image_path = os.path.join(dataset_dir, i[
                'filename'])  # load correspending image for loaded json object and save it's size
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # set jsons's 'filename' property as unique identifier since its same as original image name
            # using only 'tumor' class/object
            self.add_image('tumor', image_id=i['filename'], width=width, height=height, path=image_path,
                           polygons=polygons)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        This function loads the image from a file with given image_id
        """
        info = self.image_info[image_id]
        fp = info['path']
        image = cv2.imread(fp)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    # info = self.image_info[image_id]
    # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
    # image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
    # image = image * bg_color.astype(np.uint8)
    # for shape, color, dims in info['shapes']:
    #     image = self.draw_shape(image, shape, dims, color)
    # return image

    def load_mask(self, image_id):
        """Return instance masks for brain image of the given ID.
        """
        # If not a tumor dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tumor":
            return super(self.__class__).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(image_info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

if __name__ == "__main__":
    config = TrainConfig()
    config.display()

    DATASET_DIR = './brain-tumor-segmentation/brain_tumor_data/'
    # Train dataset generator
    dataset_train = TumorDataset()
    dataset_train.load_brain_tumor_images(DATASET_DIR,'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TumorDataset()
    dataset_val.load_brain_tumor_images(DATASET_DIR,'val')
    dataset_val.prepare()

    # Test dataset
    dataset_test = TumorDataset()
    dataset_test.load_brain_tumor_images(DATASET_DIR,'test')
    dataset_test.prepare()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')