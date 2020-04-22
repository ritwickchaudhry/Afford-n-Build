import numpy as np
from scipy.io import loadmat

from config.config import cfg

def get_filtered_indices(data):
    indices = []
    for idx, scene in enumerate(data):
        objects = scene[10]
        num_objs = objects.shape[1]
        if num_objs == 0:
            continue
        num_class_objects = 0
        objects = objects.squeeze(0) # num_objs
        for obj in objects:
            label = obj[3][0]
            if label in classes:
                num_class_objects += 1
        if num_class_objects >= cfg['MIN_NUM'] and num_class_objects <= cfg['MAX_NUM']:
            indices.append(idx)
    return np.array(indices)