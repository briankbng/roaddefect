from PIL import Image
import cv2
from yolo.yolov5 import YoloV5
yolo = YoloV5()

r_image, pred_dict = yolo.detect_image('yolo/datasets/test1/Czech_000396.jpg')
#r_image, pred_dict = yolo.detect_image('yolo/datasets/test1/Czech_000809.jpg')
print("yolov5_rdd -> pred_dict is ", pred_dict)

image = Image.fromarray(r_image)
image.show()

#cv2.imshow("predicted image", r_image)
#cv2.waitKey(3000)  # 3 seconds
