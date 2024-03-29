import os
import math
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from matplotlib.path import Path
import matplotlib.pyplot as plt
import itertools
from collections import Counter

from torch.utils.data import Dataset
from data.filter import get_filtered_indices

from shapely.geometry import Polygon

from config.config import cfg
from src.geom_transforms import compute_eligibility, shuffle_scene, get_total_extents, scale_boxes, get_extents_of_boxes, is_contained
from data.make_random_configurations import make_random_configuration
from src.utils import Tree

from torchvision.transforms import Compose
from data.transforms import RandomFlip, RandomRotation, MakeSquare

class SUNRGBD(Dataset):
	_classes = cfg['CLASSES']

	def __init__(self, data_root, cache_dir, data=None, split="train"):
		self.data_root = data_root
		self.cache_dir = cache_dir
		self.split = split
		if data is not None:
			self.data = data
		else:
			self.data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
		self.label_to_index = {label:index for index,label in enumerate(SUNRGBD._classes)}
		self.get_bboxdb()
		self.transform = Compose([
			RandomFlip(p=0.5),
			RandomFlip(p=0.5, direction='vertical'),
			RandomRotation(-30,30),
			MakeSquare()
		])

	@property
	def classes(self):
		return SUNRGBD._classes
	
	def image_path_at(self, i):
		imgpath = os.path.join(self.data_root, *self.data[i][4][0].split('/')[5:])
		return imgpath

	def add_oriented_box(self, image, box, label):
		'''
			Takes in a map image, and an oriented box,
			and appends the box to the map image
		'''
		H, W = image.shape
		poly_path = Path(box)
		y, x = np.mgrid[:H, :W]
		coords = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
		mask = poly_path.contains_points(coords).reshape(H,W)
		image[mask] = label
		return image
	
	def add_oriented_stack(self, image, box, label, height):
		C, H, W = image.shape
		poly_path = Path(box)
		y, x = np.mgrid[:H, :W]
		coords = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
		mask = poly_path.contains_points(coords).reshape(H,W)
		image[label, mask] = height
		return image

	@staticmethod
	def add_masked_oriented_stack(image, box, label, tier):
		C, H, W = image.shape
		poly_path = Path(box)
		y, x = np.mgrid[:H, :W]
		coords = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
		mask = poly_path.contains_points(coords).reshape(H,W)
		# NOTE: Can change this to incorporate multiple objects
		# of the same class by incrementing the mask
		image[label, mask] = np.maximum(image[label, mask], tier)
		return image
	
	@staticmethod
	def gen_masked_stack(boxes, labels, tiers, extents=None, sz=None):
		if extents is None:
			x_min, x_max, y_min, y_max = get_extents_of_boxes(boxes)
			extents = (x_max-x_min, y_max-y_min)
		num_classes = len(SUNRGBD._classes)
		boxes, H, W = scale_boxes(boxes, extents, sz=sz)
		image = np.zeros((num_classes, H, W), dtype=np.float)	# One channel for each class
		for box, label, tier in zip(boxes, labels, tiers):
			image = SUNRGBD.add_masked_oriented_stack(image, box[:,:2], label, tier)
		return image

	@staticmethod
	def viz_map_image(image):
		plt.imshow(image, cmap='jet')
		plt.show()
	
	@staticmethod
	def viz_pair_map_images(image1, image2):
		fig, ax = plt.subplots(1,2)
		ax[0].imshow(image1, cmap='jet')
		ax[1].imshow(image2, cmap='jet')
		plt.show()

	def gen_map(self, boxes, labels):
		'''
			Takes in a list of oriented boxes,
			and generates a map image
		'''
		boxes, H, W, x_min, x_max, y_min, y_max= SUNRGBD.scale_boxes(boxes)
		# image = np.zeros((cfg['H'], cfg['W']), dtype=np.uint8)
		image = np.zeros((H, W), dtype=np.uint8)
		for box, label in zip(boxes, labels):
			image = self.add_oriented_box(image, box[:,:2], label)
		npad = ((cfg['PAD']),(cfg['PAD']))
		image = np.pad(image, npad, 'constant', constant_values=0)
		return (x_min, x_max, y_min, y_max), image

	@staticmethod
	def convert_masked_stack_to_map(image):
		'''
			Takes in a stack image (C, H, W) and converts it to a visual
			map image
		'''
		C, H, W = image.shape
		assert C == len(SUNRGBD._classes), "Incorrect dims"

		masks = (image > 0) * 1.0
		map_image = np.argmax(image, axis=0) + 1
		obj_absent_mask = np.sum(masks, axis=0) == 0
		map_image[obj_absent_mask] = 0.0
		cm = plt.get_cmap('jet', lut=len(SUNRGBD._classes)+1)
		coloured_map = cm(map_image)
		coloured_map = (coloured_map[:, :, :3] * 255).astype(np.uint8)
		return Image.fromarray(coloured_map)

	# OLDMETHOD
	def convert_masked_stack_to_height(self, image):
		'''
			Takes in a stack image (C, H, W) and converts it to a visual
			height encoded image
		'''
		C, H, W = image.shape
		assert C == 2*len(self.classes), "Incorrect dims"

		heights = image[1::2]
		height_image = np.max(heights*255.0, axis=0)
		return height_image

	def get_bboxdb(self):
		# cache_path = os.path.join(self.cache_dir, 'bboxdb_{}.pkl'.format(self.split))
		# if os.path.exists(cache_path):
		# 	print('Loading bboxdb from cached file')
		# 	self.img_corner_list = pickle.load(open(cache_path, 'rb'))
		# 	return

		total_eligible_scenes, total_scenes = np.array([0,0,0,0,0]), 0
		self.img_corner_list = []
		for scene_idx,scene in enumerate(tqdm(self.data)):
			corners_list = []
			label_list = []
			area_list = []
			height_list = []
			
			objects = scene[10]
			num_class_objects = 0
			objects = objects.squeeze(0)
			for obj in objects:
				label = obj[3][0]
				if label in SUNRGBD._classes:
					num_class_objects += 1
			if num_class_objects < cfg['MIN_NUM'] or num_class_objects > cfg['MAX_NUM']:
				continue

			for obj in objects:
				label = obj[3][0]
				if label not in SUNRGBD._classes:
					continue
				basis = obj[0] * obj[1].T
				corner_1 = obj[2] + basis[0] + basis[1]
				corner_2 = obj[2] - basis[0] + basis[1]
				corner_3 = obj[2] - basis[0] - basis[1]
				corner_4 = obj[2] + basis[0] - basis[1]
				
				height = obj[2].squeeze(0)[2]

				corners_list.append(np.vstack([corner_1, corner_2, corner_3, corner_4]))
				label_list.append(self.label_to_index[obj[3][0]])
				area_list.append(np.linalg.norm(basis[0]) * np.linalg.norm(basis[1]))
				height_list.append(height)
			
			corners_list = np.stack(corners_list)
			label_list = np.stack(label_list)

			# Create tree of objects and get tiers
			tree = Tree(corners_list, label_list, height_list)
			tiers = np.array([min(node.tier/2, 1.0) for node in tree])
			
			
			self.img_corner_list.append({'vertices': corners_list, 'labels': label_list, 'areas': np.array(area_list),
											'heights': height_list, 'tiers': tiers})
			elgibilities, _, _ = compute_eligibility(corners_list, max_pad=5)
			total_eligible_scenes += np.array([int(x) for x in elgibilities])
			total_scenes += 1

		# if not os.path.exists(self.cache_dir):
		# 	os.mkdir(self.cache_dir)
		# pickle.dump(self.img_corner_list, open(cache_path, "wb"))
		print('Total :{}, Eligible: {}'.format(total_scenes, total_eligible_scenes))
		return
	
	def get_IoU(self, corners_1, corners_2):
		box_1 = Polygon([corner[:2] for corner in corners_1])
		box_2 = Polygon([corner[:2] for corner in corners_2])
		IoU = box_1.intersect(box_2) / box_1.union(box_2)
		
		return IoU

	def __len__(self):
		return len(self.img_corner_list)

	def __getitem__(self, index):
		bboxes = self.img_corner_list[index]['vertices']
		labels = self.img_corner_list[index]['labels']
		heights = self.img_corner_list[index]['heights']
		tiers = self.img_corner_list[index]['tiers']
		areas = self.img_corner_list[index]['areas']
		
		shuffled_boxes, shuffled_tiers = shuffle_scene(bboxes[:,:,:2].copy(), areas.copy())
		extents = get_total_extents(bboxes, shuffled_boxes)

		image = SUNRGBD.gen_masked_stack(bboxes, labels, tiers, extents)
		random_image = SUNRGBD.gen_masked_stack(shuffled_boxes, labels, shuffled_tiers, extents)

		# -----------------------------------------------------------
		# ---------------------- VISUALIZATION ----------------------
		# -----------------------------------------------------------	
		map_image = SUNRGBD.convert_masked_stack_to_map(image)
		random_map_image = SUNRGBD.convert_masked_stack_to_map(random_image)
		SUNRGBD.viz_pair_map_images(map_image, random_map_image)
		# -----------------------------------------------------------

		# image - num_classes x H x W
		image = self.transform(image)
		random_image = self.transform(random_image)
		
		return image, random_image

if __name__ == '__main__':
	data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
	# import pdb; pdb.set_trace()
	filtered_indices = get_filtered_indices(data)
	train_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data=data[filtered_indices], split="train")
	import pdb; pdb.set_trace()
	# for i in range(4):
		# train_dataset[i]
	# import pdb; pdb.set_trace()
	# for img_id, scene in enumerate(train_dat)
	# import pdb; pdb.set_trace()
	# train_dataset[0]
	# val_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_val], split='val')
	# data_obj[0]

	#################################
	# data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
	# filtered_indices = get_filtered_indices(data)
	# train_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data=data[filtered_indices], split="train")
	# stats = Counter()
	# img_stats = {}
	# for img_id, scene in enumerate(train_dataset.img_corner_list):
	# 	num_boxes = len(scene['vertices'])
	# 	boxes = scene['vertices']
	# 	labels = scene['labels']
	# 	for i, j in itertools.permutations(range(num_boxes), 2):
	# 		if is_contained(boxes[i], boxes[j]):
	# 			key = (SUNRGBD._classes[labels[i]], SUNRGBD._classes[labels[j]])
	# 			if key == ('pillow', 'bed'):
	# 				import pdb; pdb.set_trace()
	# 				train_dataset[img_id]
	# 			stats[key] += 1
	# 			img_list = img_stats.setdefault(key, [])
	# 			img_list.append(train_dataset.image_path_at(img_id))
	# print(stats)
	# import pdb; pdb.set_trace()