import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import rotate

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
<<<<<<< HEAD
		# NOTE: Assumed - Image Shape - (C,H,W)
		dtype, device = image.dtype, image.device
		image = image.numpy()
		H, W = image.shape[1:]
		randomCrops = np.zeros(((self.numCrops,image.shape[0]) + self.outputSize), dtype=image.dtype)
		for i in range(self.numCrops):
			top = np.random.randint(0, H - self.outputSize[0])
			left = np.random.randint(0, W - self.outputSize[1])

			randomCrops[i,...] = image[:, top : top + self.outputSize[0],
									left : left + self.outputSize[0]]
		return F.interpolate(torch.tensor(randomCrops, dtype=dtype, device=device), size=self.scaleSize)

if __name__ == '__main__':
	image = np.arange(3*20*20).reshape((3,20,20))
	padder = Pad()
	import pdb; pdb.set_trace()
=======
		angle = self.sample_angle()
		image = rotate(image, angle, axes=(1,2), mode='constant', cval=0.0, order=0)
		return image
>>>>>>> 7b5c8ff2bc34211886d3817dd0ed5c733aa40147
