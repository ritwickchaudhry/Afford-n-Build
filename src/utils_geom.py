import numpy as np

def translation_mat(tx, ty):
    mat = np.eye(3)
    mat[0,2] = tx
    mat[1,2] = ty
    return mat

def rotation_mat(angle, units='deg'):
    mat = np.eye(3)
    if units == 'deg':
        angle = angle * np.pi/180.
    mat[0,0] = np.cos(angle)
    mat[0,1] = -np.sin(angle)
    mat[1,0] = np.sin(angle)
    mat[1,1] = np.cos(angle)
    return mat

def apply_transform(points, T):
    assert len(points.shape) == 2 and points.shape[1] == 2, "Shape: {}".format(points.shape)
    points_ = np.vstack((points.T,np.ones(points.shape[0])))
    transformed_points_ = T @ points_
    transformed_points = (transformed_points_/transformed_points_[2])[:2]
    return transformed_points.T