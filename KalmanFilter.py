import numpy as np

a = 0.0005
var_a = 0.01
var_x = 10
var_y = 10
var_w = 5
var_h = 5

class KalmanFilter:
  def __init__(self, _id: int, x: float, y: float, x_vel: float, y_vel: float, bbox_w, bbox_h):
    self.id = _id
    self.x = np.array([
      [x],
      [y],
      [x_vel],
      [y_vel],
      [bbox_w],
      [bbox_h]
    ])
    self.P = np.eye(6)
    self.age = 0

  def age_self(self):
    self.age += 1

  @property
  def pos(self):
    return (self.x[0][0], self.x[1][0])

  @property
  def bbox(self):
    return (self.x[4][0], self.x[5][0])

  def predict(self, dt=1):
    F = np.array([  
      [1, 0, dt, 0, 0, 0],
      [0, 1, 0, dt, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1]
    ])

    B = np.array([
      [dt**2 / 2],
      [dt**2 / 2],
      [dt],
      [dt],
      [5],
      [5]
    ])

    Q = np.array([
      [dt**4/4, 0, dt**3/2, 0, 0, 0],
      [0, dt**4/4, 0, dt**3/2, 0, 0],
      [dt**3/2, 0, dt**2, 0, 0, 0],
      [0, dt**3/2, 0, dt**2, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1],
    ])

    # state prediction
    global a
    x_new = F.dot(self.x) + B * a
    global var_a
    P_new = F.dot(self.P).dot(F.T) + Q * var_a

    self.x = x_new
    self.P = P_new
    return None

  def update(self, x_meas, y_meas, bbox_w, bbox_h):
    self.age = 0
    z = np.array([
      [x_meas],
      [y_meas],
      [bbox_w],
      [bbox_h]
    ])

    H = np.array([
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1],
    ])

    R = np.array([
      [var_x, 0, 0, 0],
      [0, var_y, 0, 0],
      [0, 0, var_w, 0],
      [0, 0, 0, var_h]
    ])

    # measurement prediction
    z_pred = H.dot(self.x)
   
    S = H.dot(self.P).dot(H.T) + R
    K = self.P.dot(H.T).dot(np.linalg.inv(S))

    x_est = self.x + K.dot(z - z_pred)
    I = np.eye(6)
    P_est = (I - K.dot(H)).dot(self.P)

    self.x = x_est
    self.P = P_est

  def __str__(self):
    return str(self.pos)