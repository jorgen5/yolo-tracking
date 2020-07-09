import cv2
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

    for kalman in kalmans:
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
          kalmans.remove(kalman)
      else:
        closest_det = dists[0][0]
        centroid = closest_det.centroid
        kalman.update(centroid[0], centroid[1])
        kalman.matched_det = closest_det
        available_detections.remove(closest_det)

    # left-over (probably new) detections
    for det in available_detections:
      print("UUS")
      if has_kalman(det, kalmans): #?????
        continue
      kalman = KalmanFilter(index, det.centroid[0], det.centroid[1], 0., 0.)
      kalman.matched_det = det
      index += 1
      kalmans.append(kalman)
  return kalmans

def has_kalman(det, kalmans):
  for kalman in kalmans:
    dist = euclidean_distance(kalman.pos, det.centroid)
    print(dist)
    if dist < 20:
      return True
  return False

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
      cv2.rectangle(frame, k_start_point, k_end_point, (0, 255, 0), 2)

      #cv2.circle(frame, det.centroid, 4, (0, 0, 255), -1)
      cv2.circle(frame, kalman_centroid, 4, (0, 255, 0), -1)

      id_pos = (k_start_point[0] - 5, k_start_point[1] - 5)
      cv2.putText(
        frame,
        str(id_), 
        id_pos, 
        font, 
        fontScale,
        (255, 255, 255),
        lineType)

      """
      speed = math.sqrt(kalman.x[2][0]**2 + kalman.x[3][0]**2) * 5
      speed_pos = (k_end_point[0] - 15, k_start_point[1] - 5)
      cv2.putText(
        frame,
        str(int(speed)), 
        speed_pos,
        font, 
        0.4,
        (0, 255, 0),
        1)
      """

    if cv2.waitKey(33) == ord('q'):
      break
    
    cv2.imshow('frame', frame)
    out.write(frame)
    cv2.waitKey(1)

  capture.release()
  out.release()
  cv2.destroyAllWindows()

track_cars("data3.mp4")