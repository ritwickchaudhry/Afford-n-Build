import torch
import torchvision
import numpy as np
import torch.nn.functional as F

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

if __name__ == '__main__':
	image = np.arange(3*20*20).reshape((3,20,20))
	padder = Pad()
	import pdb; pdb.set_trace()