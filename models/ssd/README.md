# SSD Development for Road Defect Detector

## Introduction

This submodule is trying to implement Object Detection for the defected road via [SSD(Single Shot Multiple Detector)](https://arxiv.org/abs/1512.02325)

As building SSD from scratch would be a challenging and time-consuming task, current common practice is using Transfer Learning based on other 
well-trained model. 

My task for this project is to implement SSD via [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
currently in TensorFlow implementation, SSD has following pre-trained model: 

* [SSD MobileNet v2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)
* [SSD MobileNet v1 FPN 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz)
* [SSD MobileNet V2 FPNLite 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)
* [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)
* [SSD ResNet50 V1 FPN 640x640 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)
* [SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)
* [SSD ResNet101 V1 FPN 640x640 (RetinaNet101)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz)
* [SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)
* [SSD ResNet152 V1 FPN 640x640 (RetinaNet152)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz)
* [SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)

In my experiment, I have tried the `SSD MobileNet v2 320x320` and the `SSD ResNet50 V1 FPN 640x640`, as the experiment shows with current dataset, with default hyperparameter
`SSD ResNet50 V1 FPN 640x640` have issue of overfitting, lower inference speed, and much larger model size, taking our project further implementation into mobile devices. 
`SSD MobileNet v2 320x320` was chosen. 


## For first time user

As current [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
was not yet published in `conda` or `pip`, this API need to build from source in order to walk through my submodule. 

### Step 1 Proper directory structure
In current directory, we need to create such subdirectory:

```python
import os


# for the pre-trained model
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


# a proper path setup
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }


# frequent used files' path
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# if subdirectory have not created, create it
for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            !mkdir -p {path}
        if os.name == 'nt':
            !mkdir {path}
```

### Step 2 `git clone` the TensorFlow Object Detection API from TensorFlow GitHub repo

* `git clone` the repo as follow:
```python
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    !git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}
```

* for `macOS` user, there's dependency need to be done externally
  * Install [Homebrew](https://brew.sh) if you haven't, you should do when turn on your mac for the first time
  * `brew install protobuf` to install [Protocol Buffers](https://developers.google.com/protocol-buffers)

* for `linux (ubuntu)` user, install it this way:
  * `apt-get update`
  * `apt-get install protobuf-compiler`

* for `Windows` user, please follow my training notebook in this directory for step `0` -- `1`

### Step 3, install the TensorFlow Object Detection API and run through the test

For this step, please run through my notebook step `0` -- `1`

**For inference, you are good to go till this step, I strongly recommend to run through my `SSD_MobileNet` notebook step `0` -- `1` for inference**


## To inference as class

There's a class called `SSDMobileNetDetector` in `ssd_detector.py` ready to use, example: 

In the project `src` directory, or you have install `roaddefectdetecor` as package via `setup.py` in project root directory, simply:

```python
from src.SSD_LU_zhiping.ssd_detector import SSDMobileNetDetector

image_path = 'Tensorflow/workspace/images/train/Japan_000012.jpg'

detector = SSDMobileNetDetector()
detection = detector.detection(image_path=image_path)

>>> detection
Out[5]: 
{'num_instances': 4,
 'image_height': 600,
 'image_width': 600,
 'Boxes': [[94.52686607837677,
   255.95437288284302,
   395.10977268218994,
   586.885142326355],
  [402.7637243270874, 600.0, 457.3630928993225, 598.7996220588684],
  [0.0, 164.11921977996826, 364.52057361602783, 545.9090709686279],
  [289.83263969421387, 600.0, 303.7167549133301, 538.7379884719849]],
 'scores': [1.0, 1.0, 1.0, 0.97],
 'pred_classes': ['D20', 'D20', 'D44', 'D44']}
```
