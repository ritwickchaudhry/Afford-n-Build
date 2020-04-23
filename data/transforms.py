import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import rotate
from scipy.ndimage import zoom

from config.config import cfg

class Pad(object):
	"""
		Adds constant padding to the image boundaries
		Args:
			pad_size (tuple or int): Desired padding. If int, equal padding is used in both dimensions
	"""
	def __init__(self, pad_size=None):
		if pad_size is not None:
			assert isinstance(pad_size, (int, tuple))
			if isinstance(pad_size, int):
				self.pad_size = ((0,0), (pad_size,pad_size), (pad_size,pad_size))
			else:
				assert len(pad_size) == 2
				self.pad_size = ((0,0), (pad_size[0],pad_size[0]), (pad_size[1],pad_size[1]))
		else:
			self.pad_size = ((0,0),(cfg['PAD'],cfg['PAD']),(cfg['PAD'],cfg['PAD']))
	
	def __call__(self, image):
		# NOTE: Assumed - Image Shape - (C,H,W)
		image = np.pad(image, self.pad_size, 'constant', constant_values=0)
		return image

class MakeSquare(object):
	"""
		Adds padding to the make the image square
	"""
	def __init__(self):
		pass
	
	def __call__(self, image):
		# NOTE: Assumed - Image Shape - (C,H,W)
		_, H, W = image.shape
		
		max_dim = max(H, W)
		scale = cfg['H']/max_dim
		newH, newW = int(scale*H), int(scale*W)
		image = zoom(image, (1, scale, scale), order=0, mode='constant', cval=0.0)

		_, H, W = image.shape

		padH_top = (cfg['H'] - H)//2
		padH_bottom = (cfg['H'] - H) - (cfg['H'] - H)//2
		padH = (padH_top, padH_bottom)

		padW_left = (cfg['W'] - W)//2
		padW_right = (cfg['W'] - W) - (cfg['W'] - W)//2
		padW = (padW_left, padW_right)

		pad_size = ((0,0), padH, padW)
		image = np.pad(image, pad_size, 'constant', constant_values=0)
		return image

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