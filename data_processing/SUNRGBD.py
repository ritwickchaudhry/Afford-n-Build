import os
import math
import pickle
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from matplotlib.path import Path
import matplotlib.pyplot as plt

from config import cfg
from make_random_configurations import make_random_configuration

class SUNRGBD:
	def __init__(self):
		self.data_root = '../data/'
		self.cache_dir = '../cache/'
		data_path = os.path.join(self.data_root, 'SUNRGBDMeta3DBB_v2.mat')
		self.data = loadmat(data_path)['SUNRGBDMeta'].squeeze()
		self._classes = _CLASSES =  ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 
				'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 
				'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 
				'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'ottoman', 'dresser_mirror', 'drawer')
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

	def scale_boxes(self, boxes):
		x_min, x_max = boxes[:,:,0].min(), boxes[:,:,0].max()
		y_min, y_max = boxes[:,:,1].min(), boxes[:,:,1].max()
		x_diff, y_diff = (x_max - x_min), (y_max - y_min)
		scale = cfg['H']/max(x_diff, y_diff)
		boxes[:,:,0] = boxes[:,:,0] - x_min
		boxes[:,:,1] = boxes[:,:,1] - y_min
		boxes = boxes * scale
		return boxes, math.ceil(y_diff*scale), math.ceil(x_diff*scale), x_min, x_max, y_min, y_max

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
		plt.imshow(image, cmap='jet')
		plt.show()
		return x_min, x_max, y_min, y_max

	def get_bboxdb(self):
		cache_path = os.path.join(self.cache_dir, 'bboxdb.pkl')
		if os.path.exists(cache_path):
			print('Loading bboxdb from cached file')
			self.img_corner_list = pickle.load(open(cache_path, 'rb'))
			return

		self.img_corner_list = []
		for scene in tqdm(self.data[:5]):
			corners_list = []
			label_list = []
			area_list = []
			
			objects = scene[10]
			num_objs = objects.shape[1]
			if num_objs == 0 or num_objs > cfg['MAX_NUM'] or num_objs < cfg['MIN_NUM']:
				continue
			objects = objects.squeeze(0) # num_objs

			for obj in objects:
				basis = obj[0] * obj[1].T
				corner_1 = obj[2] + basis[0] + basis[1]
				corner_2 = obj[2] - basis[0] + basis[1]
				corner_3 = obj[2] - basis[0] - basis[1]
				corner_4 = obj[2] + basis[0] - basis[1]
				
				corners_list.append(np.vstack([corner_1, corner_2, corner_3, corner_4]))
				label_list.append(self.label_to_index[obj[3][0]])
				area_list.append(np.linalg.norm(basis[0]) * np.linalg.norm(basis[1]))

			corners_list = np.stack(corners_list)
			label_list = np.stack(label_list)
			self.img_corner_list.append({'vertices' : corners_list, 'labels' : label_list, "areas": area_list})
		if not os.path.exists(self.cache_dir):
			os.mkdir(self.cache_dir)
		pickle.dump(self.img_corner_list, open(cache_path, "wb"))
		return

	def __len__(self):
		self.len(self.img_corner_list)

	def __getitem__(self, index):
		bboxes = self.img_corner_list[index]['vertices']
		labels = self.img_corner_list[index]['labels']
		print(self.image_path_at(index))
		extents = self.gen_map(bboxes, labels)
		random_bboxes = make_random_configuration(bboxes, self.img_corner_list[index]['areas'], extents)
		self.gen_map(random_bboxes, labels)


if __name__ == '__main__':
	sun = SUNRGBD()
	sun[0]
	# img = sun.get_bboxdb()