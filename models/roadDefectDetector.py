# Imports some common libraries.
import os
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import colorsys

import torch
import torch.nn.functional as F
import numpy as np

from ensemble_boxes import *

# Imports the Data paths.
from definitions import MODELS_DIR

# Imports the object detection models
from yolo.yolov5 import YoloV5
from efficientdet.efficientdet import Efficientdet
from faster_rcnn.road_defect_faster_rcnn import RoadDefectRCNN
from ssd.ssd_detector import SSDMobileNetDetector

# Imports a simple utility for data and image splitting.
import faster_rcnn.road_defect_dataCfg as dataCfg

ALL_MODELS_labels_list = []
ALL_MODELS_boxes_norm_list = []
ALL_MODELS_boxes_list = []
ALL_MODELS_scores_list = []

def label_to_index(label):
	labelIndex = {"D00" : 0, "D01" : 1, "D10" : 2, "D11" : 3, "D20" : 4, "D40" : 5, "D43" : 6, "D44" : 7, "D50" : 8, "D0w0" : 9}
	return labelIndex[label]

def drawBB(image, font, top_label, top_boxes, top_conf, colors, thickness, class_name):
	
	draw = ImageDraw.Draw(image)

	for i, c in list(enumerate(top_label)):

		predicted_class = c
		box = top_boxes[i]
		score = top_conf[i]
		top, left, bottom, right = box

		top = max(0, np.floor(top).astype('int32'))
		left = max(0, np.floor(left).astype('int32'))
		bottom = min(image.size[1], np.floor(bottom).astype('int32'))
		right = min(image.size[0], np.floor(right).astype('int32'))

		label = '{} {:.2f}'.format(predicted_class, score)
		label_size = draw.textsize(label, font)
		label = label.encode('utf-8')
		print(label, top, left, bottom, right)

		if top - label_size[1] >= 0:
			text_origin = np.array([left, top - label_size[1]])
		else:
			text_origin = np.array([left, top + 1])

		for t in range(thickness):
			draw.rectangle([left + t, top + t, right - t, bottom - t], outline=colors[class_name.index(c)])

		draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[class_name.index(c)])
		draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
	
	del draw

	return image


def Model_Faster_RCNN(image_file, image_rsize, image_csize):

		temp_label_list = []
		temp_box_list = []
		temp_box_norm_list = []
		temp_score_list = []

		single_label_onehot = []

		# Do image detection for faster_rcnn
		im = cv2.imread(image_file)
		image_shape = np.array(np.shape(im)[0:2])
		print(">>>> Image file name : ", d["file_name"])
		print(">>>> Image size (row, height) : ", image_shape[0], image_shape[1]) # 0 - row (height), 1 - column (width)

		r_image0, faster_rcnn_outputs = faster_rcnn.predict1(im)
		image0 = Image.fromarray(r_image0)

		draw = ImageDraw.Draw(image0)
		font = ImageFont.truetype(font='efficientdet/model_data/simhei.ttf', size=30)
		draw.text((10, 10), "RCNN Outputs", (0, 0, 0), font=font)

		image0.show()

		t = faster_rcnn_outputs["instances"].to("cpu")
		print(t.get_fields())
		print(">>>> Num of instances detected by faster_rcnn : ", len(t))

		for idx in range(len(t)):
			if t[idx].pred_classes.item() == 5: # D40
				single_label_onehot.append(1)
			else:
				single_label_onehot.append(0)

			score = t[idx].scores.item()
			temp_score_list.append(score)

			# Need to normalize the box
			box = t[idx].pred_boxes.tensor.numpy().tolist()[0]

			normalize_list = [  box[1]/image_csize,
								box[0]/image_rsize, 
								box[3]/image_csize,
								box[2]/image_rsize ]
			temp_box_norm_list.append(normalize_list)
			temp_box_list.append(box)

		ALL_MODELS_labels_list.append(single_label_onehot)
		ALL_MODELS_scores_list.append(temp_score_list)
		ALL_MODELS_boxes_list.append(temp_box_list)
		ALL_MODELS_boxes_norm_list.append(temp_box_norm_list)

def Model_Faster_YoloV5(image_file, image_rsize, image_csize):

		temp_label_list = []
		temp_box_list = []
		temp_box_norm_list = []
		temp_score_list = []

		single_label_onehot = []

		# Do image detection for yolov5
		r_image1, pred_dict1 = yolo.detect_image(image_file)
		image1 = Image.fromarray(r_image1)

		draw = ImageDraw.Draw(image1)
		font = ImageFont.truetype(font='efficientdet/model_data/simhei.ttf', size=30)
		draw.text((10, 10), "YOLO V5 Outputs", (0, 0, 0), font=font)

		image1.show()

		print(">>>> Num of instances detected by Yolo : ", pred_dict1['num_instances'])

		for idx in range(pred_dict1['num_instances']):
			pred_classes = pred_dict1['pred_classes'][idx]
			
			# single_label_onehot[label_to_index(pred_classes)] = 1
			if pred_classes == "D40":
				single_label_onehot.append(1)
			else:
				single_label_onehot.append(0)

			score = pred_dict1['scores'][idx]
			temp_score_list.append(score)

			# Need to normalize the box
			box = pred_dict1['pred_boxes'][idx]

			normalize_list = [  box[1]/image_rsize, 
								box[0]/image_csize,
								box[3]/image_rsize, 
								box[2]/image_csize ]

			temp_box_norm_list.append(normalize_list)
			temp_box_list.append(box)
		
		ALL_MODELS_labels_list.append(single_label_onehot)
		ALL_MODELS_scores_list.append(temp_score_list)
		ALL_MODELS_boxes_list.append(temp_box_list)
		ALL_MODELS_boxes_norm_list.append(temp_box_norm_list)

def Model_Faster_EfficientDet(image_file, image_rsize, image_csize):

		temp_label_list = []
		temp_box_list = []
		temp_box_norm_list = []
		temp_score_list = []

		single_label_onehot = []

		# Do image detection for efficientdet.
		image2 = Image.open(image_file)
		r_image2, pred_dict2 = efficientdet.detect_image(image2)

		draw = ImageDraw.Draw(r_image2)
		font = ImageFont.truetype(font='efficientdet/model_data/simhei.ttf', size=30)
		draw.text((10, 10), "EfficientDet Outputs", (0, 0, 0), font=font)

		r_image2.show()
		
		print(">>>> Num of instances detected by efficientDet : ", pred_dict2['num_instances'])

		for idx in range(pred_dict2['num_instances']):
			pred_classes = pred_dict2['pred_classes'][idx]
			# single_label_onehot[label_to_index(pred_classes)] = 1
			if pred_classes == "D40":
				single_label_onehot.append(1)
			else:
				single_label_onehot.append(0)

			score = pred_dict2['scores'][idx]
			temp_score_list.append(score)

			# Need to normalize the box
			box = pred_dict2['pred_boxes'][idx]

			normalize_list = [  box[0]/image_rsize, 
								box[1]/image_csize,
								box[2]/image_rsize, 
								box[3]/image_csize ]

			temp_box_norm_list.append(normalize_list)
			temp_box_list.append(box)

		
		ALL_MODELS_labels_list.append(single_label_onehot)
		ALL_MODELS_scores_list.append(temp_score_list)
		ALL_MODELS_boxes_list.append(temp_box_list)
		ALL_MODELS_boxes_norm_list.append(temp_box_norm_list)

def Model_SSD(image_file, image_rsize, image_csize):

		temp_label_list = []
		temp_box_list = []
		temp_box_norm_list = []
		temp_score_list = []

		single_label_onehot = []

		# Do image detection for efficientdet.
		pred_dict2 = ssd.detection(image_file)
		
		print(">>>> Num of instances detected by SSD : ", pred_dict2['num_instances'])

		for idx in range(pred_dict2['num_instances']):
			pred_classes = pred_dict2['pred_classes'][idx]
			# single_label_onehot[label_to_index(pred_classes)] = 1
			if pred_classes == "D40":
				single_label_onehot.append(1)
			else:
				single_label_onehot.append(0)

			score = pred_dict2['scores'][idx]
			temp_score_list.append(score)

			# Need to normalize the box
			box = pred_dict2['pred_boxes'][idx]

			normalize_list = [  box[0]/image_rsize, 
								box[1]/image_csize,
								box[2]/image_rsize, 
								box[3]/image_csize ]

			temp_box_norm_list.append(normalize_list)
			temp_box_list.append(box)

		
		ALL_MODELS_labels_list.append(single_label_onehot)
		ALL_MODELS_scores_list.append(temp_score_list)
		ALL_MODELS_boxes_list.append(temp_box_list)
		ALL_MODELS_boxes_norm_list.append(temp_box_norm_list)

def model_results_fusion(weights, image_rsize, image_csize, fusion_type='nms', iou_thr=0.5, skip_box_thr=0.0001, sigma=0.1):

	scores = []
	labels = []

	if fusion_type == 'nms':
		boxes, scores, labels = nms( ALL_MODELS_boxes_norm_list,
				   					 ALL_MODELS_scores_list,
				   					 ALL_MODELS_labels_list, 
				   					 weights=weights, 
				   					 iou_thr=iou_thr )
	elif fusion_type == 'soft_nms':

		boxes, scores, labels = soft_nms( ALL_MODELS_boxes_norm_list, 
										  ALL_MODELS_scores_list, 
										  ALL_MODELS_labels_list, 
										  weights=weights, 
										  iou_thr=iou_thr, 
										  sigma=sigma,
										  thresh=skip_box_thr )


	elif fusion_type == 'non_max_weighted':
		boxes, scores, labels = non_maximum_weighted( ALL_MODELS_boxes_norm_list, 
													 ALL_MODELS_scores_list, 
													 ALL_MODELS_labels_list, 
													 weights=weights, 
													 iou_thr=iou_thr, 
													 skip_box_thr=skip_box_thr )

	elif fusion_type == 'weighted':

		boxes, scores, labels = weighted_boxes_fusion( ALL_MODELS_boxes_norm_list,
                                                       ALL_MODELS_scores_list,
		                                               ALL_MODELS_labels_list,
		                                               weights=weights,
		                                               iou_thr=iou_thr,
		                                               skip_box_thr=skip_box_thr )


	new_boxes = []
	for box in boxes:
		box = [ box[0]*image_rsize, 
				box[1]*image_csize,
				box[2]*image_rsize, 
				box[3]*image_csize ]
		new_boxes.append(box)

	return new_boxes, scores, labels 



if __name__ == "__main__":

	# Instantiate all the models
	yolo = YoloV5()
	efficientdet = Efficientdet()
	faster_rcnn = RoadDefectRCNN(os.getcwd(), 'R_50_FPN_3x', 0.2)
	ssd = SSDMobileNetDetector()

	# Use the image splits information from faster_cnn models
	# splits_per_dataset = ( "ltest/India", "ltest/Japan", "ltest/Czech")
	splits_per_dataset = ( "ltest/Japan",)
	dataset_dicts = dataCfg.load_images_ann_dicts(dataCfg.ROADDEFECT_DATASET, splits_per_dataset, dataCfg.RDD_DEFECT_CATEGORIES_ALL)

	for idx, d in enumerate(random.sample(dataset_dicts, 1)):

		im = cv2.imread(d["file_name"])
		image_shape = np.array(np.shape(im)[0:2])
		input_image = Image.fromarray(im)
		input_image.show()

		Model_Faster_RCNN(d["file_name"], image_shape[0], image_shape[1])

		Model_Faster_YoloV5(d["file_name"], image_shape[0], image_shape[1])

		Model_Faster_EfficientDet(d["file_name"], image_shape[0], image_shape[1])

		# Model_SSD(d["file_name"], image_shape[0], image_shape[1])

		print("................")

		weights = [1, 1, 1]
		iou_thr = 0.5
		skip_box_thr = 0.0001
		sigma = 0.1
		fusion_type = 'weighted'

		new_boxes, scores, labels = model_results_fusion(weights, image_shape[0], image_shape[1], fusion_type, iou_thr, skip_box_thr, sigma)

		num_classes = 4
		class_name = ['--', 'D20', 'D20', 'D40']
		hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
		colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
		myFont = ImageFont.truetype(font='efficientdet/model_data/simhei.ttf', size=np.floor(3e-2 * input_image.size[1] + 0.5).astype('int32'))
		thickness = int(max((input_image.size[0] + input_image.size[1]) // 720, 1))

		m_label = [class_name[int(i)] for i in labels]
		r_image = drawBB(input_image, myFont, m_label, new_boxes, scores, colors, thickness, class_name)
		

		draw = ImageDraw.Draw(r_image)
		myFont2 = ImageFont.truetype(font='efficientdet/model_data/simhei.ttf', size=30)
		draw.text((10, 10), "COMBINED Outputs - Box fusions", (0, 0, 0), font=myFont2)
		r_image.show()

