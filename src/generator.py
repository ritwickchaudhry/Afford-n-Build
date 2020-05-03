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
import math
from src.geom_transforms import teleport, place_on_top

from src.geom_transforms import translate

class Generator():
	def __init__(self):
		super().__init__()
		self.transform = Compose([
			MakeSquare()
		])

		self.device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
		self.model = xception(num_objects=len(cfg['CLASSES']))
		self.model.to(self.device)
		
	def next(self, all_corners, tiers, extents, num_neighbours=1):
		all_new_corners = []
		all_new_tiers = []

		for n in range(num_neighbours):
			p = np.random.random_sample()
			if p < 1.0:
				new_corners, new_tiers = self.next_place_on_top(all_corners, tiers)
			else:
				new_corners, new_tiers = self.next_teleport(all_corners, tiers, extents)
			all_new_corners.append(new_corners)
			all_new_tiers.append(new_tiers)
		return np.stack(all_new_corners, axis=0), np.stack(all_new_tiers, axis=0)	# 20 x num_objects x 4 x 3
	
	def next_place_on_top(self, all_corners, tiers):
		# import pdb; pdb.set_trace()
		tier_one_objects = np.where(np.isclose(tiers, cfg['TIERS'][0]))[0]
	
		if tier_one_objects.shape[0] >= 2:
			idx1, idx2 = np.random.choice(tier_one_objects, 2, replace=False)
			new_corners, new_tiers = place_on_top(all_corners.copy(), tiers.copy(), idx1, idx2)
			return new_corners, new_tiers
		else:
			return all_corners, tiers
	
	def next_teleport(self, all_corners, tiers, extents):
		obj_index = np.random.choice(all_corners.shape[0])
		p = np.random.random_sample()
		if p < 0.5:
			dim = 0
		else:
			dim = 1
		new_corners, new_tiers = teleport(all_corners.copy(), extents, tiers.copy(), dim, obj_index)
		return new_corners, new_tiers

	@torch.no_grad()
	def score(self, images):
		self.model.eval()
		scores = F.softmax(self.model(images).view(-1), dim=-1)
		return scores.cpu().numpy()
	
	def hill_climbing(self, all_corners, labels, tiers, extents, num_neighbours=20, beam_width=5, num_steps=2):
		# TODO: Handle the new tiers that are returned by self.next(...)
		all_corners_list = all_corners[None,...]	# 1 x num_objs x 4 x 3
		all_tiers_list = tiers[None,...]			# 1 x num_objects
		top_images = []
		for step in range(num_steps):
			all_new_corners_list = []
			all_new_tiers_list = []
			for all_corners in all_corners_list:
				all_new_corners, all_new_tiers = self.next(all_corners, tiers, extents, num_neighbours)
				all_new_corners_list.append(all_new_corners)
				all_new_tiers_list.append(all_new_tiers)

			# Concatenate new and old corners and pass through model to get scores
			all_new_corners_list.append(all_corners_list)
			all_new_tiers_list.append(all_tiers_list)
			all_new_corners_list = np.concatenate(all_new_corners_list, axis=0)	# 21/105 x num_objects x 4 x 3
			all_new_tiers_list = np.concatenate(all_new_tiers_list, axis=0)		# 21/105 x num_objects

			
			image_extents = (extents[1]-extents[0], extents[3]-extents[2])
			images = [self.transform(SUNRGBD.gen_masked_stack(all_corners, labels, tiers, image_extents))
						for all_corners, tiers in zip(all_new_corners_list, all_new_tiers_list)]
			

			if step == 0:
				SUNRGBD.viz_pair_map_images(SUNRGBD.convert_masked_stack_to_map(images[-1]),
											SUNRGBD.convert_masked_stack_to_map(images[0]))
				SUNRGBD.viz_pair_map_images(SUNRGBD.convert_masked_stack_to_map(images[-1]),
											SUNRGBD.convert_masked_stack_to_map(images[0]))
				top_images.append(SUNRGBD.convert_masked_stack_to_map(images[-1]))

			images = torch.Tensor(np.stack(images, axis=0)).to(self.device)
			scores = self.score(images)
			
			# Pick top beam-width number of configurations w/o replacement
			top_indices = np.random.choice(images.shape[0], beam_width, replace=False, p=scores)
			all_corners_list = all_new_corners_list[top_indices]
			all_tiers_list = all_new_tiers_list[top_indices]

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
	generator.hill_climbing(np.array(corners), np.array(labels), np.array(tiers), np.array(extents), num_steps=5)


	#####
	# from src.test import *
	# image = SUNRGBD.gen_masked_stack(bboxes, labels, heights, extents)
	# map_image = SUNRGBD.convert_masked_stack_to_map(image)
	# SUNRGBD.viz_map_image(map_image)
	# print(place_on_top(all_corners3, idx1, idx2))
	# import pdb; pdb.set_trace()
	# map_image = SUNRGBD.convert_masked_stack_to_map(all_corners3)
	# SUNRGBD.viz_map_image(map_image)