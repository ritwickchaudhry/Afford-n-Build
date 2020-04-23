import torch
import torchvision
import numpy as np
import torch.nn.functional as F

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


class Rotate(object):
	"""
		Flipping the map image
		Args:
			direction: 'horizontal' or 'vertical
	"""

	def __init__(self, direction='horizontal'):
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
		image = image.transpose((1,2,0))
		image = self.fun(image)
		return image.transpose((2,0,1))

class MultipleRandomCrops(object):
	"""
	Multiple Random Crops.
	Args:
		outputSize (tuple or int): Desired output size. If int, square crop is made.
		numCrops: Number of Random Crops        
	"""

	def __init__(self, outputSize, numCrops, scaleSize):
		assert isinstance(outputSize, (int, tuple))
		if isinstance(outputSize, int):
			self.outputSize = (outputSize, outputSize)
		else:
			assert len(outputSize) == 2
			self.outputSize = outputSize
		assert isinstance(numCrops, int)
		self.numCrops = numCrops
		assert isinstance(scaleSize, int)
		self.scaleSize = scaleSize

	def __call__(self, image):
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