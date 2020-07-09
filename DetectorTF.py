import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from Detection import Detection
import sys
from absl import app, flags, logging
from absl.flags import FLAGS
FLAGS(sys.argv)

class DetectorTF:
  def __init__(self, weights, class_names=["car"]):
    self.class_names = class_names
    yolo = YoloV3(classes=1)
    yolo.load_weights(weights)
    self.yolo = yolo

  def get_detections(self, image):
    width = image.shape[1]
    height = image.shape[0]

    img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    boxes, scores, classes, nums = self.yolo.predict(img_in)
 
    detections = []
    for box in boxes[0]:
      if all(x == 0. for x in box):
        continue
      x0, y0, x1, y1 = box
      start_point = (int(x0 * width), int(y0 * height))
      end_point = (int(x1 * width), int(y1 * height))
      
      centroid = (start_point[0] + int((end_point[0] - start_point[0]) / 2), start_point[1] + int((end_point[1] - start_point[1]) / 2))

      detections.append(Detection(0, start_point, end_point, centroid))
      
    return detections