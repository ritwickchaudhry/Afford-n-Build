import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

from config import cfg

data_root = '../data/SUNRGBDMeta3DBB_v2.mat'

def get_corners():
	img_corner_list = []
	data = loadmat(data_root)['SUNRGBDMeta'].squeeze()

	for scene in tqdm(data):
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
			# import pdb; pdb.set_trace()
			area_list.append(np.linalg.norm(basis[0]) * np.linalg.norm(basis[1]))

		corners_list = np.stack(corners_list)
		label_list = np.stack(label_list)    
		img_corner_list.append({'vertices' : corners_list, 'labels' : label_list, "areas": area_list})
	return img_corner_list

	
if __name__ == '__main__':
	get_corners()