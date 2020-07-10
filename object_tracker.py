import cv2
import numpy as np
import math
from KalmanFilter import KalmanFilter
from DetectorTF import DetectorTF

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
lineType = 2

detectorTF = DetectorTF("./yolov3-tf2/checkpoints/yolov3_3.tf")

def euclidean_distance(c1, c2):
  return math.sqrt((c2[0] - c1[0])**2 +  (c2[1] - c1[1])**2)


index = 1
def update_tracks(detections, kalmans):
  if kalmans == None:
    kalmans = []
    for det in detections:
      global index
      kalman = KalmanFilter(index, det.centroid[0], det.centroid[1], 0., 0.)
      kalman.matched_det = det
      index += 1
      kalmans.append(kalman)
  else:
    available_detections = detections.copy()

    remove_kalmans = []
    for i, kalman in enumerate(kalmans):
      old_pos = kalman.pos

      kalman.predict()
      dists = []
      for det in available_detections:
        dist = euclidean_distance(kalman.pos, det.centroid)
        dists.append((det, dist))
      dists.sort(key=lambda x: x[1])

      if len(dists) == 0 or dists[0][1] > 25:
        x, y = kalman.pos
        kalman.age_self()
        
        if kalman.age > 50:
          remove_kalmans.append(kalman)
      else:
        closest_det = dists[0][0]
        centroid = closest_det.centroid

        bbox_w = abs(closest_det.start_point[0] - closest_det.end_point[0])
        bbox_h = abs(closest_det.start_point[1] - closest_det.end_point[1])

        kalman.update(centroid[0], centroid[1])
        kalman.matched_det = closest_det
        available_detections.remove(closest_det)

      new_pos = kalman.pos
      pos_dif = np.array(new_pos) - np.array(old_pos)
    
    for k in remove_kalmans:
      kalmans.remove(k)

    # left-over (probably new) detections
    for det in available_detections:
      """
      if has_kalman(det, kalmans): #?????
        close_ones.append(det)
        print("close", det)
        for d in sorted(detections, key=lambda x: (x.centroid[0], x.centroid[1])):
          print(d, id(d))
        for k in kalmans:
          dist = euclidean_distance(k.pos, det.centroid)
          if dist < 20:
            print("=================VALE=================")
            print(k.matched_det, id(k.matched_det), k.age)
            for dk in k.dets:
              print(dk)
        continue
      """
      kalman = KalmanFilter(index, det.centroid[0], det.centroid[1], 0., 0.)
      kalman.matched_det = det
      index += 1
      kalmans.append(kalman)
  return kalmans

"""
def has_kalman(det, kalmans):
  for kalman in kalmans:
    dist = euclidean_distance(kalman.pos, det.centroid)
    if dist < 20:
      return True
  return False
"""

def track_cars(video):
  capture = cv2.VideoCapture(video)
  width  = int(capture.get(3))
  height = int(capture.get(4))
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
  kalmans = None
  while(True):
    _, frame = capture.read()

    detections = detectorTF.get_detections(frame)

    kalmans = update_tracks(detections, kalmans)

    for kalman in kalmans:
      id_ = kalman.id
      det = kalman.matched_det
      kalman_centroid = (int(kalman.pos[0]), int(kalman.pos[1]))

      rect_w = abs(det.start_point[0] - det.end_point[0])
      rect_h = abs(det.start_point[1] - det.end_point[1])

      # shifting the bbox to the kalman centroid
      k_start_point = (int(kalman_centroid[0] - rect_w / 2), int(kalman_centroid[1] - rect_h / 2))
      k_end_point = (int(kalman_centroid[0] + rect_w / 2), int(kalman_centroid[1] + rect_h / 2))
      cv2.rectangle(frame, k_start_point, k_end_point, (252, 78, 78), 2)

      #cv2.circle(frame, det.centroid, 4, (0, 0, 255), -1)
      #cv2.circle(frame, kalman_centroid, 4, (0, 255, 0), -1)
      
      speed = int(math.sqrt(kalman.x[2][0]**2 + kalman.x[3][0]**2) * 7.76)
      if speed > 0:
        cv2.rectangle(frame, (k_start_point[0], k_start_point[1]), (k_start_point[0] + 30, k_start_point[1] - 20), (252, 78, 78), -1)
        id_pos = (k_start_point[0] + 3, k_start_point[1] - 5)
        cv2.putText(
          frame,
          str(speed), 
          id_pos, 
          font, 
          0.6,
          (229, 208, 188),
          2)

    if cv2.waitKey(33) == ord('q'):
      break
    
    cv2.imshow('frame', frame)
    out.write(frame)
    cv2.waitKey(40)

  capture.release()
  out.release()
  cv2.destroyAllWindows()

track_cars("data3.mp4")