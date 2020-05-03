import numpy as np
import torch
import torch.nn.functional as F
from archs.xception import xception
from PIL import Image
from torchvision.transforms import Compose
from data.transforms import RandomFlip, RandomRotation, MakeSquare
from data.SUNRGBD import SUNRGBD
from config.config import cfg
import matplotlib.pyplot as plt
from src.geom_transforms import teleport, place_on_top

from src.geom_transforms import translate

class Generator():
	def __init__(self):
		super().__init__()
		self.transform = Compose([
			MakeSquare()
		])

		self.device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
		self.model = xception(num_objects=len(cfg['CLASSES'] * 2))
		self.model.to(self.device)
		
	def next(self, all_corners, extents, num_neighbours=1):
		all_new_corners = []

		for n in range(num_neighbours):
			p = np.random.random_sample()
			if p < 1.0:
				new_corners = self.next_place_on_top(all_corners)
			else:
				new_corners = self.next_teleport(all_corners, extents)
			all_new_corners.append(new_corners)
		return np.stack(all_new_corners, axis=0)	# 20 x num_objects x 4 x 3
	
	def next_place_on_top(self, all_corners):
		idx1, idx2 = np.random.choice(all_corners.shape[0], 2, replace=False)
		new_corners = place_on_top(all_corners.copy(), idx1, idx2)
		return new_corners
	
	def next_teleport(self, all_corners, extents):
		obj_index = np.random.choice(all_corners.shape[0])
		p = np.random.random_sample()
		if p < 0.5:
			dim = 0
		else:
			dim = 1
		new_corners = teleport(all_corners.copy(), extents, dim, obj_index)
		return new_corners

	@torch.no_grad()
	def score(self, images):
		self.model.eval()
		scores = F.softmax(self.model(images).view(-1), dim=-1)
		return scores.cpu().numpy()
	
	def hill_climbing(self, all_corners, labels, heights, extents, num_neighbours=20, beam_width=5, num_steps=2):
		all_corners_list = all_corners[None, :, :, :]	# 1 x num_objs x 4 x 3
		top_images = []
		for step in range(num_steps):
			all_new_corners_list = []
			for all_corners in all_corners_list:
				all_new_corners_list.append(self.next(all_corners, extents, num_neighbours))	# 20 x num_objects x 4 x 3


			# Concatenate new and old corners and pass through model to get scores
			all_new_corners_list.append(all_corners_list)
			all_new_corners_list = np.concatenate(all_new_corners_list, axis=0)	# 21/105 x num_objects x 4 x 3
			
			image_extents = (extents[1]-extents[0], extents[3]-extents[2])
			images = [self.transform(SUNRGBD.gen_masked_stack(all_corners, labels, heights, image_extents))
						for all_corners in all_new_corners_list]
			
			if step == 0:
				# SUNRGBD.viz_map_image(SUNRGBD.convert_masked_stack_to_map(images[0]))
				# SUNRGBD.viz_map_image(SUNRGBD.convert_masked_stack_to_map(images[-1]))
				top_images.append(SUNRGBD.convert_masked_stack_to_map(images[-1]))
			
			images = torch.Tensor(np.stack(images, axis=0)).to(self.device)
			scores = self.score(images)
			
			# Pick top beam-width number of configurations w/o replacement
			top_indices = np.random.choice(images.shape[0], beam_width, replace=False, p=scores)
			all_corners_list = np.stack(all_new_corners_list, axis=0)[top_indices]
			# Save top image in gif
			top_image = images[top_indices[np.argmax(scores[top_indices])]].cpu().numpy()
			top_image = SUNRGBD.convert_masked_stack_to_map(top_image)	# 128 x 128
			top_images.append(top_image)
		
		print(top_images)
		print("===================")
		top_images[0].save("0.png")
		top_images[0].save('out.gif', save_all=True, append_images=top_images[1:], fps=0.1, loop=0, optimize=False)
		
		return all_corners_list

if __name__ == '__main__':
	# from src.test import *
	# generator = Generator()
	# print(generator.get_teleportation_extents(all_corners2, extents2, 1, index2))
	from src.test_hill_climbing import *
	generator = Generator()
	generator.hill_climbing(np.array(corners), np.array(labels), np.array(heights), np.array(extents), num_steps=5)


	#####
	# from src.test import *
	# image = SUNRGBD.gen_masked_stack(bboxes, labels, heights, extents)
	# map_image = SUNRGBD.convert_masked_stack_to_map(image)
	# SUNRGBD.viz_map_image(map_image)
	# print(place_on_top(all_corners3, idx1, idx2))
	# import pdb; pdb.set_trace()
	# map_image = SUNRGBD.convert_masked_stack_to_map(all_corners3)
	# SUNRGBD.viz_map_image(map_image)