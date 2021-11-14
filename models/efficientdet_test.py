from PIL import Image
from efficientdet.efficientdet import Efficientdet

efficientdet = Efficientdet()
image = Image.open('efficientdet/img/Japan_013109.jpg')
image.show()
r_image, pred_dict = efficientdet.detect_image(image)
r_image.show()
print(pred_dict)