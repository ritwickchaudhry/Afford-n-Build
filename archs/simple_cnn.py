import math
import unittest
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch


class SimpleCNN(nn.Module):
	def __init__(self, num_objects=20, num_classes=1000):
		super(SimpleCNN, self).__init__()
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(num_objects, 32, 3, 2, 0, bias=False)
		self.bn1 = nn.BatchNorm2d(32)

		self.conv2 = nn.Conv2d(32,64,3,bias=False)
		self.bn2 = nn.BatchNorm2d(64)

		self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
		self.block2=Block(128,512,2,2,start_with_relu=True,grow_first=True)
		# self.block3=Block(256,512,2,2,start_with_relu=True,grow_first=True)

		self.fc1 = nn.Sequential(
			nn.Linear(512, 128),
			nn.Dropout(p=0.4),
			nn.LeakyReLU(inplace=True)
		)

		self.fc2 = nn.Sequential(
			nn.Linear(128, 1),
			nn.Tanh()
		)

		#------- init weights --------
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
		#----------------------------
		
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x) # 32 x 111 x 111
		x = self.bn1(x) # 32 x 111 x 111
		x = self.relu(x) # 32 x 111 x 111
		
		x = self.conv2(x) # 64 x 109 x 109
		x = self.bn2(x) # 64 x 109 x 109
		x = self.relu(x) # 64 x 109 x 109
		
		x = self.block1(x) # 128 x 55 x 55
		x = self.block2(x) # 256 x 28 x 28
		# x = self.block3(x) # 512 x 14 x 14

		x = F.adaptive_avg_pool2d(x, (1,1)) # 512 x 1 x 1 - Can downsize more I think, but let's try this
		x = torch.flatten(x, start_dim=1) # 512
		x = self.fc1(x) # 128
		x = self.fc2(x) # 1

		return x

class Block(nn.Module):
	def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
		super(Block, self).__init__()

		if out_filters != in_filters or strides!=1:
			self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
			self.skipbn = nn.BatchNorm2d(out_filters)
		else:
			self.skip=None
		
		self.relu = nn.ReLU(inplace=True)
		rep=[]

		filters=in_filters
		if grow_first:
			rep.append(self.relu)
			rep.append(nn.Conv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(out_filters))
			filters = out_filters

		for i in range(reps-1):
			rep.append(self.relu)
			rep.append(nn.Conv2d(filters,filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(filters))
		
		if not grow_first:
			rep.append(self.relu)
			rep.append(nn.Conv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(out_filters))

		if not start_with_relu:
			rep = rep[1:]
		else:
			rep[0] = nn.ReLU(inplace=False)

		if strides != 1:
			rep.append(nn.MaxPool2d(3,strides,1))
		self.rep = nn.Sequential(*rep)

	def forward(self,inp):
		x = self.rep(inp)

		if self.skip is not None:
			skip = self.skip(inp)
			skip = self.skipbn(skip)
		else:
			skip = inp

		x+=skip
		return x


def simple_cnn(**kwargs):
	"""
	Construct Xception.
	"""
	model = SimpleCNN(**kwargs)
	return model


def test():
	kwargs = {
	'num_classes': 20
	}
	net = simple_cnn(**kwargs,)
	random_input = torch.rand((1,20,224,224))
	out = net(random_input)
	assert out.shape == (1,1)


if __name__ == '__main__':
	test()