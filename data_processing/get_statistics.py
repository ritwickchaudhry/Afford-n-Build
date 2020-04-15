import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

from config import cfg

data_root = '/Users/ritwickchaudhry/Downloads/SUNRGBDMeta3DBB_v2.mat'

def get_stats():
	x_list = []
	y_list = []
	z_list = []
	data = loadmat(data_root)['SUNRGBDMeta'].squeeze()
	for scene in tqdm(data):
		objects = scene[10]
		num_objs = objects.shape[1]
		if num_objs == 0 or num_objs > cfg['MAX_NUM'] or num_objs < cfg['MIN_NUM']:
			continue
		objects = objects.squeeze(0) # num_objs
		for obj in objects:
			x,y,z = obj[2].squeeze()
			x_list.append(x)
			y_list.append(y)
			z_list.append(z)
	x_, y_, z_  = [np.array(t) for t in [x_list, y_list, z_list]]
	import pdb; pdb.set_trace()

	
if __name__ == '__main__':
	get_stats()
