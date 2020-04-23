import torch
import torchvision
import numpy as np
import torch.nn.functional as F

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