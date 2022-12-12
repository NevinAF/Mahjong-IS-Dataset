import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import pickle
import random

import cv2
import google.colab.patches as cv2_imshow

#
# Define constants
#

CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
CHECKPOINT_URL = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

OUTPUT_DIR = "output"
NUM_CLASSES = 1

TRAIN_DATASET_NAME = "train"
TRAIN_DATASET_PATH = "train"
TRAIN_DATASET_ANNOTATION_PATH = "train.json"

TEST_DATASET_NAME = "test"
TEST_DATASET_PATH = "test"
TEST_DATASET_ANNOTATION_PATH = "test.json"

DEVICE_NAME = "cuda"

#
# Define util functions
#

def plot_samples(dataset_name, n=1):
	dataset_custom = DatasetCatalog.get(dataset_name)
	metadata_custom = MetadataCatalog.get(dataset_name)

	for d in random.sample(dataset_custom, n):
		img = cv2.imread(d["file_name"])
		visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_custom, scale=0.5)
		vis = visualizer.draw_dataset_dict(d)
		cv2_imshow(vis.get_image()[:, :, ::-1])

def get_train_cfg():
	cfg = get_cfg()

	cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CHECKPOINT_URL)
	cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
	cfg.DATASETS.TEST = (TEST_DATASET_NAME,)

	cfg.DATALOADER.NUM_WORKERS = 2

	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.00025
	cfg.SOLVER.MAX_ITER = 3000
	cfg.SOLVER.STEPS = []

	cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
	cfg.MODEL.DEVICE = DEVICE_NAME

	cfg.OUTPUT_DIR = OUTPUT_DIR

	return cfg

#
# Register the dataset
#

register_coco_instances(TRAIN_DATASET_NAME, {}, TRAIN_DATASET_ANNOTATION_PATH, TRAIN_DATASET_PATH)

register_coco_instances(TEST_DATASET_NAME, {}, TEST_DATASET_ANNOTATION_PATH, TEST_DATASET_PATH)

#
# Plot samples
#
print("Train dataset samples")
plot_samples(TRAIN_DATASET_NAME, 2)
print("Test dataset samples")
plot_samples(TEST_DATASET_NAME, 2)

#
# Train the model
#

cfg = get_train_cfg()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

with open(os.path.join(cfg.OUTPUT_DIR, "config.pkl"), "wb") as f:
	pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

trainer.train()