import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import os
import pickle
from config import cfg

data_root = '../data/'
cache_dir = '../cache/'

class SUNRGBD:

    def __init__(self):
        data_path = os.path.join(data_root, 'SUNRGBDMeta3DBB_v2.mat')
        self.data = loadmat(data_path)['SUNRGBDMeta'].squeeze()
        self._classes = _CLASSES =  ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 
                'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 
                'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 
                'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag')
    
    @property
    def classes(self):
        return self._classes
    
    def image_path_at(self, i):
        imgpath = os.path.join(data_root, *self.data[i][4][0].split('/')[5:])
        return imgpath

    def get_bboxdb(self):
        cache_path = os.path.join(cache_dir, 'bboxdb.pkl')
        if os.path.exists(cache_path):
            print('Loading bboxdb from cached file')
            img_corner_list = pickle.load(open(cache_path, 'rb'))
            return img_corner_list
        
        img_corner_list = []
        for scene in tqdm(self.data):
            corners_list = []
            label_list = []
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
                
                label = obj[3][0]
                corners = np.vstack([corner_1, corner_2, corner_3, corner_4])
                corners_list.append(corners)
                label_list.append(label)

            corners_list = np.stack(corners_list)
            label_list = np.stack(label_list)    
            img_corner_list.append({'vertices' : corners_list, 'labels' : label_list})
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        pickle.dump(img_corner_list, open(cache_path, "wb"))
        return img_corner_list

if __name__ == '__main__':
    sun = SUNRGBD()
    img = sun.get_bboxdb()