import numpy as np
# from tqdm import tqdm
# from scipy.io import loadmat

# from get_corners import get_corners


def make_random_configuration(all_corners, areas, extents):
	random_pos = []
	all_new_corners = []
	X_MIN, X_MAX, Y_MIN, Y_MAX = extents
	for corners in all_corners:
		radius = np.linalg.norm(corners[0] - corners[2]) / 2
		x_min = X_MIN + radius
		x_max = X_MAX - radius
		y_min = Y_MIN + radius
		y_max = Y_MAX - radius

		x_pos = np.random.uniform(low=x_min, high=x_max)
		y_pos = np.random.uniform(low=y_min, high=y_max)
		pos = np.array([x_pos, y_pos, corners[0][2]])
		angle = np.random.uniform(low=0, high = 2 * np.pi)
		x_dir = np.cos(angle)
		y_dir = np.sin(angle)
		
		len_1 =	np.linalg.norm(corners[0] - corners[1])
		len_2 =	np.linalg.norm(corners[1] - corners[2])
		dir_1 = np.array([x_dir, y_dir, 0]) * len_1
		dir_2 = np.array([y_dir, -x_dir, 0]) * len_2
		
		corner_1 = pos + dir_1 / 2 + dir_2 / 2
		corner_2 = pos - dir_1 / 2 + dir_2 / 2
		corner_3 = pos - dir_1 / 2 - dir_2 / 2
		corner_4 = pos + dir_1 / 2 - dir_2 / 2

		all_new_corners.append(np.stack([corner_1, corner_2, corner_3, corner_4]))
		# import pdb; pdb.set_trace()
	
	return np.stack(all_new_corners)

	
# if __name__ == '__main__':
# 	data = get_corners()
# 	make_random_configuration(data[0]["vertices"], data[0]["areas"])