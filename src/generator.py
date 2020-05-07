import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose
from config.config import cfg
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pointplot as pplot
from scipy.io import loadmat
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
from copy import deepcopy
from sklearn.model_selection import train_test_split

from archs.xception import xception
from src.geom_transforms import translate, shuffle_scene, teleport, place_on_top, rotate, get_extents
from src.utils import Tree
from data.filter import get_filtered_indices
from data.transforms import RandomFlip, RandomRotation, MakeSquare
from data.SUNRGBD import SUNRGBD

class Generator():
	def __init__(self):
		super().__init__()
		self.transform = Compose([
			MakeSquare()
		])

		self.device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
		self.model = xception(num_objects=len(cfg['CLASSES']))
		self.model.load_state_dict(torch.load(cfg["best_model_path"])["params"])
		self.model.to(self.device)
		
	def next(self, all_corners, tiers, extents, num_neighbours=1):
		all_new_corners = []
		all_new_tiers = []

		next_fns = [self.next_place_on_top, self.next_teleport, self.next_rotate]

		for n in range(num_neighbours):
			next_idx = np.random.choice(len(next_fns), p=cfg['next_probs'])
			new_corners, new_tiers = next_fns[next_idx](all_corners, tiers, extents)
			new_extents = get_extents(new_corners)
			if new_extents[0] < extents[0] or new_extents[2] < extents[2] or new_extents[1] > extents[1] or new_extents[3] > extents[3]:
				import pdb; pdb.set_trace()
			all_new_corners.append(new_corners)
			all_new_tiers.append(new_tiers)
		return np.stack(all_new_corners, axis=0), np.stack(all_new_tiers, axis=0)	# 20 x num_objects x 4 x 3
	
	def next_place_on_top(self, all_corners, tiers, extents):
		tier_one_objects = np.where(np.isclose(tiers, cfg['TIERS'][0]))[0]
	
		if tier_one_objects.shape[0] >= 2:
			idx1, idx2 = np.random.choice(tier_one_objects, 2, replace=False)
			new_corners, new_tiers = place_on_top(all_corners.copy(), tiers.copy(), idx1, idx2)
			return new_corners, new_tiers
		else:
			return all_corners, tiers
	
	def next_teleport(self, all_corners, tiers, extents):
		obj_index = np.random.choice(all_corners.shape[0])
		p = np.random.random_sample()
		if p < 0.5:
			dim = 0
		else:
			dim = 1
		new_corners, new_tiers = teleport(all_corners.copy(), extents, tiers.copy(), dim, obj_index)
		return new_corners, new_tiers

	def next_rotate(self, all_corners, tiers, extents):
		tier_one_objects = np.where(np.isclose(tiers, cfg['TIERS'][0]))[0]
		if tier_one_objects.shape[0] > 0:
			obj_index = np.random.choice(tier_one_objects)
			new_obj_corners, was_successful = rotate(all_corners, extents, obj_index, 90)
			if was_successful:
				all_corners[obj_index] = new_obj_corners
		return all_corners, tiers

	@torch.no_grad()
	def score(self, images):
		self.model.eval()
		scores = self.model(images).view(-1)
		normalized_score = F.softmax(scores / cfg['temperature'], dim=-1)
		return scores.cpu().numpy(), normalized_score.cpu().numpy()
	
	def hill_climbing(self, init_corners, labels, init_tiers, extents, index, num_neighbours=20, beam_width=5,
						num_steps=75, dirname="results/"):
		# TODO: Handle the new tiers that are returned by self.next(...)
		corners_list_beam = deepcopy(init_corners[None,...])	# 1 x num_objs x 4 x 3
		tiers_list_beam = deepcopy(init_tiers[None,...])			# 1 x num_objects
		top_images = []
		beam_scores = []

		for step in tqdm(range(num_steps)):
			all_new_corners_list = []
			all_new_tiers_list = []
			# For each scene in the beam, get the specified number of neighbours and their respective tiers
			for all_corners, all_tiers in zip(corners_list_beam, tiers_list_beam):
				all_new_corners, all_new_tiers = self.next(deepcopy(all_corners), deepcopy(all_tiers), extents, num_neighbours)
				all_new_corners_list.append(all_new_corners)
				all_new_tiers_list.append(all_new_tiers)

			# Append existing beam to new scenes and new tiers and concatenate the numpy arrays
			all_new_corners_list.append(deepcopy(corners_list_beam))
			all_new_tiers_list.append(deepcopy(tiers_list_beam))
			all_new_corners_list = np.concatenate(all_new_corners_list, axis=0)	# 21/105 x num_objects x 4 x 3
			all_new_tiers_list = np.concatenate(all_new_tiers_list, axis=0)		# 21/105 x num_objects

			# Get the images for all the scenes
			image_extents = (extents[1]-extents[0], extents[3]-extents[2])
			images = [self.transform(SUNRGBD.gen_masked_stack(all_corners, labels, tiers, image_extents))
						for all_corners, tiers in zip(all_new_corners_list, all_new_tiers_list)]
			
			# Append initial scene to top_images to show progress
			if step == 0:
				top_images.append(SUNRGBD.convert_masked_stack_to_map(SUNRGBD.gen_masked_stack(all_new_corners_list[-1],
																								labels,
																								all_new_tiers_list[-1],
																								image_extents,
																								sz=1024)))
				top_images[0].save(dirname + "{}_init.jpg".format(index))


			# Pass images through the model and get scores
			images = torch.Tensor(np.stack(images, axis=0)).to(self.device)
			scores, normalized_scores = self.score(images)

			if step == 0:
				beam_scores.append(np.array([scores[-1]]))
			
			# Pick top beam-width number of configurations w/o replacement - stochastic selection
			top_indices = np.random.choice(normalized_scores.shape[0], beam_width, replace=False, p=normalized_scores)
			# Hard selection
			# top_indices = np.flip(np.argsort(scores))[:beam_width]

			# Update beam
			corners_list_beam = all_new_corners_list[top_indices]
			tiers_list_beam = all_new_tiers_list[top_indices]
			beam_scores.append(scores[top_indices])

			# Save top image in gif
			top_image = images[top_indices[np.argmax(scores[top_indices])]].cpu().numpy()
			top_image_corners = all_new_corners_list[top_indices[np.argmax(scores[top_indices])]]
			top_image_tier = all_new_tiers_list[top_indices[np.argmax(scores[top_indices])]]
			top_image = SUNRGBD.convert_masked_stack_to_map(SUNRGBD.gen_masked_stack(top_image_corners,
																					labels,
																					top_image_tier,
																					image_extents,
																					sz=1024))
			# top_image = SUNRGBD.convert_masked_stack_to_map(top_image)	# 128 x 128
			top_images.append(top_image)

		
		top_images[0].save(dirname + '{}_out.gif'.format(index), save_all=True, append_images=top_images[1:], fps=0.05, loop=0, optimize=False)
		top_images[-1].save(dirname + '{}_final.jpg'.format(index))
		
		return corners_list_beam, tiers_list_beam, beam_scores 

def initialize(dataset, index):
	bboxes = dataset.img_corner_list[index]['vertices']
	areas = dataset.img_corner_list[index]['areas']
	heights = dataset.img_corner_list[index]['heights']
	labels = dataset.img_corner_list[index]['labels']
	shuffled_boxes, shuffled_tiers = shuffle_scene(bboxes[:,:,:2].copy(), areas.copy())
	extents = get_extents(shuffled_boxes)
	
	return shuffled_boxes, labels, shuffled_tiers, extents

def plot_scores_with_errorbars(scores, scene_index, dirname="results/"):
	mean_scores = np.array([np.mean(score_list) for score_list in scores])
	max_scores = np.array([np.max(score_list) for score_list in scores])
	min_scores = np.array([np.min(score_list) for score_list in scores])
	X = np.arange(len(mean_scores))
	plt.figure()
	plt.plot(X, mean_scores)
	plt.fill_between(X, min_scores, max_scores, color='b', alpha=.1)
	plt.xlabel("step")
	plt.ylabel("score")
	plt.savefig(dirname + "{}.jpg".format(scene_index))
	plt.close()
			

if __name__ == '__main__':
	generator = Generator()

	data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
	filtered_indices = get_filtered_indices(data)
	idx_trainval, idx_test = train_test_split(filtered_indices, test_size=0.1, random_state=1)
	idx_train, idx_val = train_test_split(idx_trainval, test_size=0.3, random_state=1)
	val_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_val], split='val')

	# indices = np.random.choice(len(val_dataset), 100, replace=False)
	indices = [53]
	dirname=""
	for index in indices:
		corners, labels, tiers, extents = initialize(val_dataset, index)
		print("Index {}: Labels - {}, Tiers - {}".format(index, labels, tiers))
		_, _, beam_scores = generator.hill_climbing(corners[...,:2], labels, tiers, extents, index, dirname=dirname)
		plot_scores_with_errorbars(beam_scores, index, dirname=dirname)

	# from src.test import *
	# generator = Generator()
	# print(generator.get_teleportation_extents(all_corners2, extents2, 1, index2))
	# from src.test_hill_climbing import *
	# generator = Generator()
	# generator.hill_climbing(np.array(corners), np.array(labels), np.array(tiers), np.array(extents), num_steps=5)


	#####
	# from src.test import *
	# image = SUNRGBD.gen_masked_stack(bboxes, labels, heights, extents)
	# map_image = SUNRGBD.convert_masked_stack_to_map(image)
	# SUNRGBD.viz_map_image(map_image)
	# print(place_on_top(all_corners3, idx1, idx2))
	# import pdb; pdb.set_trace()
	# map_image = SUNRGBD.convert_masked_stack_to_map(all_corners3)
	# SUNRGBD.viz_map_image(map_image)