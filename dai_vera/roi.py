import numpy as np

class ROI:
    def __init__(self, x, y, z, slice_data, time_series = None):
        self.x = x
        self.y = y
        self.z = z
        self.slice_data = slice_data    # 2d slice at selected z
        self.time_series = time_series or [] 
        # placeholders for additional metrices
        self.radius = None
        self.area = None
