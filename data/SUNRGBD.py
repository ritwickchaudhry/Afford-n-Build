import os
import math
import pickle
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from matplotlib.path import Path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from shapely.geometry import Polygon

from config.config import cfg
from data.make_random_configurations import make_random_configuration

class SUNRGBD(Dataset):
	def __init__(self, data_root, cache_dir, data=None, split="train"):
		self.data_root = data_root
		self.cache_dir = cache_dir
		self.split = split
		if data is not None:
			self.data = data
		else:
			self.data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
		# self._classes = _CLASSES =  ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 
		# 		'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 
		# 		'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 
		# 		'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'ottoman', 'dresser_mirror', 'drawer')
		self._classes = cfg['CLASSES']
		self.label_to_index = {label:index for index,label in enumerate(self._classes)}
		self.get_bboxdb()

	@property
	def classes(self):
		return self._classes
	
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

	def add_masked_oriented_stack(self, image, box, label, height):
		C, H, W = image.shape
		poly_path = Path(box)
		y, x = np.mgrid[:H, :W]
		coords = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
		mask = poly_path.contains_points(coords).reshape(H,W)
		# NOTE: Can change this to incorporate multiple objects
		# of the same class by incrementing the mask
		image[2*label, mask] = 1.0
		image[2*label+1, mask] = height
		return image

	def scale_boxes(self, boxes):
		x_min, x_max = boxes[:,:,0].min(), boxes[:,:,0].max()
		y_min, y_max = boxes[:,:,1].min(), boxes[:,:,1].max()
		x_diff, y_diff = (x_max - x_min), (y_max - y_min)
		scale = cfg['H']/max(x_diff, y_diff)
		boxes[:,:,0] = boxes[:,:,0] - x_min
		boxes[:,:,1] = boxes[:,:,1] - y_min
		boxes = boxes * scale
		return boxes, math.ceil(y_diff*scale), math.ceil(x_diff*scale), x_min, x_max, y_min, y_max
	
	def gen_stack(self, boxes, labels, heights):
		num_classes = len(self.classes)
		boxes, H, W, x_min, x_max, y_min, y_max = self.scale_boxes(boxes)
		image = np.zeros((num_classes, H, W), dtype=np.uint8)
		h_min = np.min(heights)
		h_max = np.max(heights)
		for box, label, height in zip(boxes, labels, heights):
			rescaled_height = (height-h_min)/(h_max-h_min)
			image = self.add_oriented_stack(image, box[:,:2], label, rescaled_height)
		return (x_min, x_max, y_min, y_max), image

	def gen_masked_stack(self, boxes, labels, heights):
		num_classes = len(self.classes)
		boxes, H, W, x_min, x_max, y_min, y_max = self.scale_boxes(boxes)
		image = np.zeros((2*num_classes, H, W), dtype=np.float) # Even channels - masks
		h_min = np.min(heights)
		h_max = np.max(heights)
		for box, label, height in zip(boxes, labels, heights):
			rescaled_height = (height-h_min)/(h_max-h_min)
			rescaled_height = rescaled_height/2.0 + 0.5 # Keep ht from [0.5 - 1]
			image = self.add_masked_oriented_stack(image, box[:,:2], label, rescaled_height)
		return (x_min, x_max, y_min, y_max), image

	def viz_map_image(self, image):
		plt.imshow(image, cmap='jet')
		plt.show()

	def gen_map(self, boxes, labels):
		'''
			Takes in a list of oriented boxes,
			and generates a map image
		'''
		boxes, H, W, x_min, x_max, y_min, y_max= self.scale_boxes(boxes)
		# image = np.zeros((cfg['H'], cfg['W']), dtype=np.uint8)
		image = np.zeros((H, W), dtype=np.uint8)
		for box, label in zip(boxes, labels):
			image = self.add_oriented_box(image, box[:,:2], label)
		npad = ((cfg['PAD']),(cfg['PAD']))
		image = np.pad(image, npad, 'constant', constant_values=0)
		return (x_min, x_max, y_min, y_max), image

	def convert_masked_stack_to_map(self, image):
		'''
			Takes in a stack image (C, H, W) and converts it to a visual
			map image
		'''
		C, H, W = image.shape
		assert C == 2*len(self.classes), "Incorrect dims"

		masks = image[::2]
		heights = image[1::2]
		map_image = np.argmax(heights, axis=0) + 1
		obj_absent_mask = np.sum(masks, axis=0) == 0
		map_image[obj_absent_mask] = 0.0
		return map_image

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
		cache_path = os.path.join(self.cache_dir, 'bboxdb_{}.pkl'.format(self.split))
		if os.path.exists(cache_path):
			print('Loading bboxdb from cached file')
			self.img_corner_list = pickle.load(open(cache_path, 'rb'))
			return

		self.img_corner_list = []
		for scene in tqdm(self.data[:10]):
			corners_list = []
			label_list = []
			area_list = []
			height_list = []
			
			objects = scene[10]
			num_class_objects = 0
			objects = objects.squeeze(0)
			for obj in objects:
				label = obj[3][0]
				if label in self._classes:
					num_class_objects += 1
			if num_class_objects < cfg['MIN_NUM'] or num_class_objects > cfg['MAX_NUM']:
				continue

			for obj in objects:
				label = obj[3][0]
				if label not in self._classes:
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
			self.img_corner_list.append({'vertices': corners_list, 'labels': label_list, "areas": area_list, "heights": height_list})
		
		if not os.path.exists(self.cache_dir):
			os.mkdir(self.cache_dir)
		pickle.dump(self.img_corner_list, open(cache_path, "wb"))
		return
	
	def get_IoU(self, corners_1, corners_2):
		box_1 = Polygon([corner[:2] for corner in corners_1])
		box_2 = Polygon([corner[:2] for corner in corners_2])
		IoU = box_1.intersect(box_2) / box_1.union(box_2)
		
		return IoU

	def process(self, image, neg=False):
		pass

	def __len__(self):
		return len(self.img_corner_list)

	def __getitem__(self, index):
		bboxes = self.img_corner_list[index]['vertices']
		labels = self.img_corner_list[index]['labels']
		heights = self.img_corner_list[index]['heights']
		
		extents, image = self.gen_masked_stack(bboxes, labels, heights)
		map_image = self.convert_masked_stack_to_map(image)
		self.viz_map_image(map_image)
		
		height_image = self.convert_masked_stack_to_height(image)
		self.viz_map_image(height_image)
		
		random_bboxes = make_random_configuration(bboxes, self.img_corner_list[index]['areas'], extents)
		_, random_image = self.gen_stack(random_bboxes, labels, heights)

		# image - num_classes x H x W
		image = self.process(image)
		random_image = self.process(random_image, neg=True)
		
		return image, random_image


if __name__ == '__main__':
	data_obj = SUNRGBD(cfg['data_root'], cfg['cache_dir'])
	data_obj[0]