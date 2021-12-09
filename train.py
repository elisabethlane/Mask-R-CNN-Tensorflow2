import os
import xml.etree
import xml.etree.ElementTree 
from numpy import zeros, asarray

import utils
import config
import model

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf


class TDIDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "beat")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) > 1225:
                continue

            if not is_train and int(image_id) <= 1225:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('beat'))
        return masks, asarray(class_ids, dtype='int32')

class TDIConfig(config.Config):
    
    NAME = "beat_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 1225
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 404
    IMAGE_MAX_DIM = 1024
    IMAGE_CHANNEL_COUNT = 3
    LEARNING_RATE = 0.0001
    VALIDATION_STEPS = 665

train_set = TDIDataset()
train_set.load_dataset(dataset_dir="data/", is_train=True)
train_set.prepare()

valid_dataset = TDIDataset()
valid_dataset.load_dataset(dataset_dir="data/", is_train=False)
valid_dataset.prepare()

tdi_config = TDIConfig()

model = model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=tdi_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_set, 
            val_dataset=valid_dataset, 
            learning_rate=tdi_config.LEARNING_RATE, 
            epochs=100, 
            layers='heads')

model_path = 'tdi_mask_rcnn.h5'
model.keras_model.save_weights(model_path)
