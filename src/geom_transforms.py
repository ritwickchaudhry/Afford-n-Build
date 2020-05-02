import math
import itertools
import numpy as np

from src.utils import get_extents

def get_extents_of_box(box):
	# assert box.shape == (4,3)
	xs = box[:,0]
	ys = box[:,1]
	return xs.min(), xs.max(), ys.min(), ys.max()

def convert_to_dim(min_x, max_x, min_y, max_y, dim):
	assert dim in [0,1], "Wrong dim"
	if dim == 0:
		return (min_y, max_y, min_x, max_x)
	else:
		return (min_x, max_x, min_y, max_y)

def get_translation_extent(all_corners, areas, extents, dim, object_idx):
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

def translate(all_corners, areas, extents, dim, obj_index):
	d_min, d_max = get_translation_extent(all_corners, areas, extents, dim, obj_index)
	dv = np.random.uniform(-d_min, d_max)
	all_corners[obj_index,:, dim] += dv
	return all_corners


def calculate_diameter(corners):
	assert len(corners.shape) == 2
	# Remove z if it's there
	if corners.shape[1] == 3:
		corners = corners[:,:2]
	# Assuming CW or CCW order
	d = np.linalg.norm(corners[0] - corners[2])
	return float(d)

def compute_eligibility(all_corners, max_pad=4):
	pad = list(range(max_pad))
	max_d = max([calculate_diameter(box) for box in all_corners])
	min_x, max_x, min_y, max_y = get_extents(all_corners)
	num_grid_points = [(math.floor((max_x - min_x)/max_d) + p) * (math.floor((max_y - min_y)/max_d) + p) for p in pad]
	return [x >= all_corners.shape[0] for x in num_grid_points], (min_x, max_x, min_y, max_y), max_d

def get_grids(min_dim, max_dim, dia, pad):
	K = math.floor((max_dim - min_dim)/dia)
	grids = np.linspace(start=dia, stop=(K+pad)*float(dia), num=K+pad)
	probs = np.ones(K+pad)
	probs[K:] = probs[K:] * 0.01
	return grids, probs

def shuffle_scene(all_corners):
	assert len(all_corners.shape) == 3
	assert all_corners.shape[1] == 4 and all_corners.shape[2] == 2
	grid_it = lambda x,y : np.array(list(itertools.product(x,y)))
	eligibilities, extents, dia = compute_eligibility(all_corners)
	min_x, max_x, min_y, max_y = extents
	for pad, eligibility in enumerate(eligibilities):
		if eligibility:
			# pad = 0 --> d, 2d, .. Kd
			# pad = 1 --> d, 2d, .. Kd, (K+1)d
			# pad = 2 --> d, 2d, .. Kd, (K+1)d, (K+2)d
			grids_x, probs_x = get_grids(min_x, max_x, dia, pad)			
			grids_y, probs_y = get_grids(min_y, max_y, dia, pad)
			grid_points = grid_it(grids_x, grids_y) # K1 x K2
			assert all_corners.shape[0] <= grid_points.shape[0]
			probs = grid_it(probs_x, probs_y)
			probs = probs[:,0] * probs[:,1]
			probs = probs/probs.sum()
			selected_grid_indices = np.random.choice(np.arange(grid_points.shape[0]), 
													size=all_corners.shape[0], 
													replace=False, p=probs)
			selected_grid_points = grid_points[selected_grid_indices]
			centroids = (all_corners[:,0] + all_corners[:,2])/2
			displacements = centroids - selected_grid_points # N x 2
			all_corners = all_corners - displacements[:,None,:]
			break
	# Now translate around a bit for better packing
	for i in range(2*pad):
		extents = get_extents(all_corners)
		all_new_corners, new_corners = [], []
		for idx in range(all_corners.shape[0]):
			new_corners = translate(all_corners.copy(), None, extents, 0, idx)
			all_corners = translate(new_corners, None, extents, 1, idx)
	return np.array(all_corners)