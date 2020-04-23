import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.path import Path

from config import cfg

class DataGenerator():
	'''
		Class to convert SUN-RGBD data to Map data
	'''
	def __init__(self):
		self.dataroot = '../data/SUNRGBDMeta3DBB_v2.mat'
		self.data = loadmat(self.dataroot)['SUNRGBDMeta'].squeeze()
		self.label_to_id_mapping = None

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

	def gen_map(self, boxes, labels):
		'''
			Takes in a list of oriented boxes,
			and generates a map image
		'''
		
		for box, label in zip(boxes, labels):
			image = np.zeros((cfg['H'], cfg['W']), dtype=np.uint8)
			image = self.add_oriented_box(image, box, 255)
		plt.imshow(image, cmap='gray')
		plt.show()

	def convert(self):
		img_corner_list = []
		for scene in tqdm(self.data[:1]):
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
				corner_1 = obj[2] + basis[0] / 2 + basis[1] / 2
				corner_2 = obj[2] - basis[0] / 2 + basis[1] / 2
				corner_3 = obj[2] - basis[0] / 2 - basis[1] / 2
				corner_4 = obj[2] + basis[0] / 2 - basis[1] / 2
				
				corners_list.append(np.vstack([corner_1, corner_2, corner_3, corner_4]))
				label_list.append(obj[3][0])
				area_list.append(np.linalg.norm(basis[0]) * np.linalg.norm(basis[1]))

			corners_list = np.stack(corners_list)
			label_list = np.stack(label_list)    
			img_corner_list.append({'vertices' : corners_list, 'labels' : label_list, "areas": area_list})
			self.gen_map(corners_list, label_list)
		return img_corner_list


if __name__ == '__main__':
	generator = DataGenerator()
	generator.convert()
	# boxes = [[(10,20), (20,10), (120,110), (110,120)]]
	# generator.gen_map(boxes)
