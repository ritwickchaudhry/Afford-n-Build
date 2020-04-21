import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

import matplotlib.pyplot as plt

from config import cfg

data_root = '../data/SUNRGBDMeta3DBB_v2.mat'

classes = ["bathtub", "bed", "bookshelf", "box", "chair", "counter", "desk", "door", "dresser", "garbage_bin", 
 "lamp", "monitor", "night_stand", "pillow", "sink", "sofa", "table", "tv", "toilet"]
 
def stats():
	class_counts = {cls: 0 for cls in classes}
	class_occ_counts = {cls: 0 for cls in classes}
	image_occ_counts = []
	
	data = loadmat(data_root)['SUNRGBDMeta'].squeeze()
	for scene in tqdm(data):
		img_obj_count = 0
		img_class_occ = set()
		objects = scene[10]
		num_objs = objects.shape[1]
		if num_objs == 0:
			continue
		objects = objects.squeeze(0) # num_objs
		for obj in objects:
			label = obj[3][0]
			if label in classes:
				class_counts[label] += 1
				img_class_occ.add(label)
				img_obj_count += 1
		for cls in img_class_occ:
		   class_occ_counts[cls] += 1
		image_occ_counts.append(img_obj_count)
	
	return class_counts, class_occ_counts, image_occ_counts

	
if __name__ == '__main__':
	class_counts, class_occ_counts, image_occ_counts = stats()
	print(class_counts)
	print(class_occ_counts)
	plt.hist(image_occ_counts, bins=max(image_occ_counts))
	plt.show()