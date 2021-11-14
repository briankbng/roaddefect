import time

import cv2
import numpy as np
from PIL import Image
import sys
from efficientdet import Efficientdet

if __name__ == "__main__":
    efficientdet = Efficientdet()
    mode = "predict"
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval = 100
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            #img = sys.argv[1]
            print('Open image {:s}'.format(img))
            try:
                image = Image.open(img)
                image.show()
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, pred_dict = efficientdet.detect_image(image)
                r_image.show()
                break
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
