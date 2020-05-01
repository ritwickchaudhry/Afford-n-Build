import numpy as np

def get_extents_of_box(box):
    # assert box.shape == (4,3)
    xs = box[:,0]
    ys = box[:,1]
    return xs.min(), xs.max(), ys.min(), ys.max()

def convert_to_dim(min_x, max_x, min_y, max_y, dim):
    assert dim in [0,1], "Wrong dim"
    if dim == 0:
        return (min_y, max_y, min_x, max_x)
    else:
        return (min_x, max_x, min_y, max_y)


class AvgMeter():
    def __init__(self):
        self.val = 0.0
        self.cnt = 0

    def update(self, v, c):
        self.val += v
        self.cnt += c
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else:
            return self.val/self.cnt