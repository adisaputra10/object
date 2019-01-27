from imageai.Detection import ObjectDetection
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image1.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

image = Image.open('imagenew.jpg')
image.show()
    


#  658  pip install imageai-2.0.2-py3-none-any.whl
#  659  pip3 install imageai-2.0.2-py3-none-any.whl
#  660  python3.6 -V
#  661  python3.6 detection.py 
#  662  nano detection.py
#  663  python3.6 detection.py 
#  664  nano detection.py
#  665  python3.6 detection.py 
#  666  pip3 install opencv-python
#  667  python3.6 detection.py 
#  668  pip3 install keras
#  669  python3.6 detection.py 
#  670  pip3 install tensorflow
#  671  python3.6 detection.py 
#  672  pip3 install cv2
#  673  pip3 install images
#  674  pip3 install Image
#  675  python3.6 detection.py 
#  676  pip3 install matplotlib
#  677  python3.6 detection.py 