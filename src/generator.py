import numpy as np

from utils import get_extents_of_box, convert_to_dim

class Generator():
	def __init__(self):
		super().__init__()

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

	def next(self, all_corners, areas, extents):
		all_new_corners = []
		boxes = []
		X_MIN, X_MAX, Y_MIN, Y_MAX = extents


if __name__ == '__main__':
	from test import *
	generator = Generator()
	print(generator.get_translation_extent(all_corners1, None, extents1, 1, index1))