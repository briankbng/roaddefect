from ssd.ssd_detector import SSDMobileNetDetector

# image_path = 'ssd/Tensorflow/workspace/images/train/Czech_000031.jpg'
image_path = '/mnt/c/WORKING_FOLDER/PROJECTS/MTECH/GITHUB/roaddefectdetector/SystemCode/roaddefectdetector/src/data/ltrain/Czech/images/Czech_000031.jpg'


detector = SSDMobileNetDetector()
detection = detector.detection(image_path=image_path)

print(detection)

# >>> detection