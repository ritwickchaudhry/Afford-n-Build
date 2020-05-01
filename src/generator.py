import numpy as np
import torch
import torch.nn.functional as F
from archs.xception import xception
from torchvision.transforms import Compose
from data.transforms import RandomFlip, RandomRotation, MakeSquare
from data.SUNRGBD import SUNRGBD
from src.utils import get_extents_of_box, convert_to_dim
from config.config import cfg


class Generator():
	def __init__(self):
		super().__init__()
		self.transform = Compose([
			MakeSquare()
		])

		self.device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
		self.model = xception(num_objects=len(cfg['CLASSES'] * 2))
		self.model.to(self.device)

	def get_translation_extent(self, all_corners, areas, extents, dim, object_idx):
		'''
			Invariant - Assuming the MBRs of the boxes don't overlap for the input
		'''
		assert dim == 0 or dim == 1,  "Only for x and y"
		
		X_MIN, X_MAX, Y_MIN, Y_MAX = extents
		OTHER_DIM_MIN, OTHER_DIM_MAX, DIM_MIN, DIM_MAX = convert_to_dim(*extents, dim)

		other_dim = 1-dim
		# Get the MBR of the current object
		curr_object_box = all_corners[object_idx] # 4 x 3
		min_x, max_x, min_y, max_y = get_extents_of_box(curr_object_box)
		# Transform to dim space
		curr_min_other_dim, curr_max_other_dim, curr_min_dim, curr_max_dim = convert_to_dim(min_x, max_x, min_y, max_y, dim)
		# Filter the boxes to get the target boxes
		indices = np.ones(all_corners.shape[0], dtype=bool)
		indices[object_idx] = False
		other_boxes = all_corners[indices]
		other_boxes_max_other_dim = other_boxes[:,:,other_dim].max(axis=1) # N 
		other_boxes_min_other_dim = other_boxes[:,:,other_dim].min(axis=1) # N
		target_boxes = other_boxes[~np.logical_or(other_boxes_max_other_dim <= curr_min_other_dim, 
									other_boxes_min_other_dim >= curr_max_other_dim)]
		# Segregate before and after other boxes.
		# Enforce wall checks and also empty before/after
		if target_boxes.shape[0] == 0:
			before_box_indices = None
			after_box_indices = None
		else:
			before_box_indices = target_boxes[:,:,dim].max(axis=1) <= curr_min_dim
			if before_box_indices.sum() == 0:
				before_box_indices = None
			after_box_indices = target_boxes[:,:,dim].min(axis=1) >= curr_min_dim
			if after_box_indices.sum() == 0:
				after_box_indices = None
		# Get the extents
		before_extent = target_boxes[before_box_indices,:,dim].max() if before_box_indices is not None else DIM_MIN
		after_extent = target_boxes[after_box_indices,:,dim].min() if after_box_indices is not None else DIM_MAX
		return (curr_min_dim - before_extent, after_extent - curr_max_dim)

	def translate(self, all_corners, areas, extents, dim, obj_index):
		d_min, d_max = self.get_translation_extent(all_corners, areas, extents, dim, obj_index)
		dv = np.random.uniform(-d_min, d_max)
		all_corners[obj_index,:, dim] += dv
		return all_corners
		
	
	def next(self, all_corners, areas, extents, num_neighbours=1):
		all_new_corners = []

		for n in range(num_neighbours):
			obj_index = np.random.choice(all_corners.shape[0])
			
			new_corners = self.translate(all_corners.copy(), areas, extents, 0, obj_index)
			all_new_corners.append(self.translate(new_corners, areas, extents, 1, obj_index))

		return np.stack(all_new_corners, axis=0)	# 20 x num_objects x 4 x 3
	
	@torch.no_grad()
	def score(self, images):
		self.model.eval()
		scores = F.softmax(self.model(images).view(-1), dim=-1)
		return scores.cpu().numpy()
	
	def hill_climbing(self, all_corners, areas, labels, heights, extents, num_neighbours=20, beam_width=5, num_steps=2):
		all_corners_list = all_corners[None, :, :, :]	# 1 x num_objs x 4 x 3
		for step in range(num_steps):
			all_new_corners_list = []
			for all_corners in all_corners_list:
				all_new_corners_list.append(self.next(all_corners, areas, extents, num_neighbours))	# 20 x num_objects x 4 x 3

			# Concatenate new and old corners and pass through model to get scores
			all_new_corners_list.append(all_corners_list)
			all_new_corners_list = np.concatenate(all_new_corners_list, axis=0)	# 21 x num_objects x 4 x 3
			# import pdb; pdb.set_trace()
			
			images = [self.transform(SUNRGBD.gen_masked_stack(all_corners, labels, heights)[1])
						for all_corners in all_new_corners_list]
			# if step == 0:
			# 	SUNRGBD.viz_map_image(SUNRGBD.convert_masked_stack_to_map(images[0]))
			# 	SUNRGBD.viz_map_image(SUNRGBD.convert_masked_stack_to_map(images[-1]))
			images = torch.Tensor(np.stack(images, axis=0)).to(self.device)
			scores = self.score(images)
			
			# Pick top beam-width number of configurations w/o replacement
			top_indices = np.random.choice(images.shape[0], beam_width, replace=False, p=scores)
			all_corners_list = np.stack(all_new_corners_list, axis=0)[top_indices]
		
		return all_corners_list

if __name__ == '__main__':
	# from test import *
	# generator = Generator()
	# print(generator.get_translation_extent(all_corners1, None, extents1, 1, index1))
	from src.test_hill_climbing import *
	generator = Generator()
	print(generator.hill_climbing(np.array(corners), np.array(areas), np.array(labels), np.array(heights), np.array(extents)))