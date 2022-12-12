from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

import os
import pickle
import random
import cv2
import google.colab.patches as cv2_imshow

CONFIG_SAVE_PATH = "config.pkl"

with open(CONFIG_SAVE_PATH, "rb") as f:
	cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


predictor = DefaultPredictor(cfg)



TEST_DATASET_NAME = "test"
TEST_DATASET_PATH = "test"
TEST_DATASET_ANNOTATION_PATH = "test.json"
register_coco_instances(TEST_DATASET_NAME, {}, TEST_DATASET_ANNOTATION_PATH, TEST_DATASET_PATH)

# Get random image from test dataset

dataset_custom = DatasetCatalog.get(TEST_DATASET_NAME)
metadata_custom = MetadataCatalog.get(TEST_DATASET_NAME)

d = random.choice(dataset_custom)
img = cv2.imread(d["file_name"])

# Run inference

outputs = predictor(img)

# Visualize results

v = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])

