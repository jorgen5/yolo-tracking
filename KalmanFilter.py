import numpy as np

a = 0.005
var_a = 0.01
var_x = 10
var_y = 10

class KalmanFilter:
  def __init__(self, _id: int, x: float, y: float, x_vel: float, y_vel: float):
    self.id = _id
    self.x = np.array([
      [x],
      [y],
      [x_vel],
      [y_vel]
    ])
    self.P = np.eye(4)
    self.age = 0

  def age_self(self):
    self.age += 1

  @property
  def pos(self):
    return (self.x[0][0], self.x[1][0])

  def predict(self, dt=1):
    F = np.array([  
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ])

    B = np.array([
      [dt**2 / 2],
      [dt**2 / 2],
      [dt],
      [dt]
    ])

    Q = np.array([
      [dt**4/4, 0, dt**3/2, 0],
      [0, dt**4/4, 0, dt**3/2],
      [dt**3/2, 0, dt**2, 0],
      [0, dt**3/2, 0, dt**2]   
    ])

    # state prediction
    global a
    x_new = F.dot(self.x) + B * a
    global var_a
    P_new = F.dot(self.P).dot(F.T) + Q * var_a

    self.x = x_new
    self.P = P_new
    return None

  def update(self, x_meas, y_meas):
    z = np.array([
      [x_meas],
      [y_meas]
    ])

    H = np.array([
      [1, 0, 0, 0],
      [0, 1, 0, 0]
    ])

    R = np.array([
      [var_x, 0],
      [0, var_y]
    ])

    # measurement prediction
    z_pred = H.dot(self.x)
   
    S = H.dot(self.P).dot(H.T) + R
    K = self.P.dot(H.T).dot(np.linalg.inv(S))

    x_est = self.x + K.dot(z - z_pred)
    I = np.eye(4)
    P_est = (I - K.dot(H)).dot(self.P)

    self.x = x_est
    self.P = P_est

  def __str__(self):
    return str(self.x)