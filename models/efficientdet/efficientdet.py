import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from efficientdet.nets.efficientdet import EfficientDetBackbone
from efficientdet.utils.utils import (cvtColor, get_classes, image_sizes, preprocess_input,
                         resize_image)
from efficientdet.utils.utils_bbox import decodebox, non_max_suppression

class Efficientdet(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   model_path: Contain the trained model
        #   classes_path: Define the category of object
        #--------------------------------------------------------------------------#
        #"model_path"        : 'logs/ep037-loss0.340-val_loss0.344.pth',
        # "classes_path"      : 'model_data/voc_classes.txt',
        "model_path"        : 'efficientdet/logs/ep025-loss0.360-val_loss0.383.pth',
        "classes_path"      : 'efficientdet/model_data/rdd_classes.txt',

        #---------------------------------------------------------------------#
        #   Select model version，0-7
        #---------------------------------------------------------------------#
        "phi"               : 0,
        #---------------------------------------------------------------------#
        #   Confidence threshold
        #---------------------------------------------------------------------#
        "confidence"        : 0.08,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        #---------------------------------------------------------------------#
        #   Use cuda or not, for CPU configuration, set to False
        #---------------------------------------------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initializaiton
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.input_shape                    = [image_sizes[self.phi], image_sizes[self.phi]]
        #---------------------------------------------------#
        #   Get the number of category
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        
        #---------------------------------------------------#
        #   Setting different color for each category
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        
    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        #----------------------------------------#
        #   Create Efficientdet Model
        #----------------------------------------#
        self.net    = EfficientDetBackbone(self.num_classes, self.phi)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image_name(self, image_name):
        image = Image.open(image_name)
        r_image, pred_dict = self.detect_image(image)
        return r_image, pred_dict
    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   Get image width and height
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Transfer image to RGB format
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Resize
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Image preprocessing
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        #---------------------------------------------------------#
        #   Prepare the output dictionary for hybrid
        #---------------------------------------------------------#
        pred_dict = {"num_instances": 0,
                      "image_height": image_shape[0],
                      "image_width": image_shape[1],
                      "pred_boxes": [],
                      "scores": [],
                      "pred_classes": []}

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   pass the image into net for prediction
            #---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)
            
            #-----------------------------------------------------------#
            #   decoding
            #-----------------------------------------------------------#
            outputs     = decodebox(regression, anchors, self.input_shape)
            results     = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape, 
                                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
               
            if results[0] is None: 
                return image, pred_dict

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

            pred_dict["num_instances"] = len(top_label)
            pred_dict["pred_boxes"] = top_boxes.tolist()
            pred_dict["scores"] = top_conf.tolist()
            pred_dict["pred_classes"] = [self.class_names[int(x)] for x in top_label]

        #---------------------------------------------------------#
        #   set font and thickness
        #---------------------------------------------------------#
        # font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        font        = ImageFont.truetype(font='efficientdet/model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
         
        #font        = ImageFont.truetype("arial.ttf", 15)
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        #---------------------------------------------------------#
        #   draw image
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, pred_dict

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Transfer image to RGB format
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Resize
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   image pre-processing
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   pass the image into net for prediction
            #---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)
            
            #-----------------------------------------------------------#
            #   decoding
            #-----------------------------------------------------------#
            outputs     = decodebox(regression, anchors, self.input_shape)
            results     = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape, 
                                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
               
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
