import math
import itertools
import numpy as np

def get_wall_free_boxes():
	pass

def get_extents(all_corners):
	min_x, max_x = all_corners[:,:,0].min(), all_corners[:,:,0].max()
	min_y, max_y = all_corners[:,:,1].min(), all_corners[:,:,1].max()
	return [float(x) for x in [min_x, max_x, min_y, max_y]]

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