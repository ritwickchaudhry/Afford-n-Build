import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import rotate

class RandomFlip(object):
	"""
		Flipping the map image
		Args:
			direction: 'horizontal' or 'vertical
	"""

	def __init__(self, p=0.5, direction='horizontal'):
		self.p = p
		assert self.p <= 1 and self.p >= 0, "Invalid probability value"
		self.direction = direction
		assert self.is_valid_direction(), "Invalid direction"
		self.fun = self.direction_to_flip_fun()

	def is_valid_direction(self):
		return self.direction in ['horizontal', 'vertical']
	
	def direction_to_flip_fun(self):
		if self.direction == 'horizontal':
			fun = np.fliplr
		elif self.direction == 'vertical':
			fun = np.flipud
		else:
			assert False, "Wrong direction"
		return fun
	
	def __call__(self, image):
		sample = np.random.random()
		if sample <= self.p:
			image = image.transpose((1,2,0))
			image = self.fun(image)
			return image.transpose((2,0,1))
		else:
			pass
		return image


class RandomRotation(object):
	"""
		Rotating the map image, randomly
		Args:
			min: Minimum Angle to rotate by
			max: Maximum Angle to rotate by
	"""

	def __init__(self, min_angle=-20, max_angle=20):
		self.min = min_angle
		self.max = max_angle
		self.is_valid_angle()

	def is_valid_angle(self):
		assert self.max >= self.min
		assert self.min >= -180
		assert self.max <= 180
	
	def sample_angle(self):
		angle = np.random.random() * (self.max - self.min) + self.min
		return angle

	def __call__(self, image):
		angle = self.sample_angle()
		image = rotate(image, angle, axes=(1,2), mode='constant', cval=0.0, order=0)
		return image