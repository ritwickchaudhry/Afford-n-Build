import numpy as np
from shapely.geometry import box
 from shapely import affinity

def get_bounds(dim):
    return (corners[0][dim], corners[1][dim], corners[2][dim], corners[3][dim])

def make_neighbors(all_corners, areas, extents, orient_deg=10, trans_amt=5):
	random_pos = []
	all_new_corners = []
	X_MIN, X_MAX, Y_MIN, Y_MAX = extents
    room = box(X_MIN, Y_MIN, X_MAX, Y_MAX)
    x_trans = (trans_amt/100) * (X_MAX - X_MIN)
    y_trans = (trans_amt/100) * (Y_MAX - Y_MIN)


	for corners in all_corners:
        min_x = min(get_bounds(0))
        max_x = max(get_bounds(0))
        min_y =  min(get_bounds(1))
        minx =  max(get_bounds(1))

        heights = np.array(get_bounds(2))[:, None] #all points height should be the same anyway
        obj = box(min_x, min_y, max_x, max_y)

        #Rotate the object
        while(True) :
            angle = np.random.uniform(-orient_deg, orient_deg)
            rot_obj = affinity.rotate(obj, angle) 
            if rot_obj.intersection(room).area == rot_obj.area:
                break
                # No part of it lies outside the room

        #Translate the object
        while(True):
            t_x = np.random.uniform(-x_trans, x_trans)
            t_y = np.random.uniform(-y_trans, y_trans)
            trans_obj = affinity.translate(rot_obj, t_x, t_y)
            if trans_obj.intersection(room).area == trans_obj.area:
                break
                # No part of it lies outside the room

        corners = np.array(trans_obj.exterior.coords)[:4]
        corners = np.concatenate((corners, heights)), 1)  
		all_new_corners.append(corners)
		# import pdb; pdb.set_trace()
	
	return np.stack(all_new_corners)

	
# if __name__ == '__main__':
# 	data = get_corners()
# 	make_random_configuration(data[0]["vertices"], data[0]["areas"])