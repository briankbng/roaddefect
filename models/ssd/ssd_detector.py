"""
Author: Johnny(zhiping) LU
email: lu.zhiping@u.nus.edu
Project: Road Defect Detector
Summary: Return object detection array via trained TensorFlow SSD MobileNet model
"""
import os
import logging
import tensorflow as tf
import cv2
import numpy as np
import PIL.Image as Image

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import object_detection
except ModuleNotFoundError:
    raise ModuleNotFoundError("TensorFlow Object Detection API not install, \n\tPlease walk through notebook step 0 "
                              "-- 1")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711' \
                       '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz '
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('ssd','Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('ssd','Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('ssd','Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('ssd','Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('ssd','Tensorflow', 'protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join('ssd','Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


class SSDMobileNetDetector:
    def __init__(self):
        self.configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-31')).expect_partial()

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def _detect(self, image_path):
        img = cv2.imread(image_path)
        image_np = np.array(img)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64) + 1
        return detections

    def detection(self, image_path):
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        img = cv2.imread(image_path)
        image_np = np.array(img)
        detection = self._detect(image_path)
        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        im_width, im_height = image_pil.size
        return_dict = dict()
        boxes = []
        scores = []
        pred_classes = []

        box_to_display_str_map = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detection['detection_boxes'],
            detection['detection_classes'],
            detection['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)
        for key, value in box_to_display_str_map.items():
            ymin, xmin, ymax, xmax = key
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            boxes.append(np.array([left, right, top, bottom]))
            scores_classes_list = [x.strip() for x in value.pop(0).split(':')]
            scores.append(float(scores_classes_list[1].replace("%", '')) / 100)
            pred_classes.append(scores_classes_list[0])

        return_dict['num_instances'] = len(box_to_display_str_map)
        return_dict['image_height'] = im_height
        return_dict['image_width'] = im_width
        return_dict['Boxes'] = boxes
        return_dict['scores'] = scores
        return_dict['pred_classes'] = pred_classes

        return return_dict
