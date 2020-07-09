class Detection:
  def __init__(self, class_id, start_point, end_point, centroid):
    self.class_id = class_id
    self.start_point = start_point
    self.end_point = end_point
    self.centroid = centroid

  def __str__(self):
    return str(self.centroid)