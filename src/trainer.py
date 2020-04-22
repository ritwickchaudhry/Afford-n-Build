import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from data_processing.filter import get_filtered_indices

from archs.xception import xception
from data_processing.SUNRGBD import SUNRGBD
from config.config import cfg


class Trainer:
    def __init__(self):
        data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
        filtered_indices = get_filtered_indices(data)
        idx_trainval, _ = train_test_split(filtered_indices, test_size=0.1, random_state=1)
        idx_train, idx_val = train_test_split(idx_trainval, test_size=0.3, random_state=1)
        train_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_train], split='train')
        val_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_val], split='val')

        self.train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                        num_workers=cfg['num_workers'])
        self.val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                        num_workers=cfg['num_workers'])

        self.device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
        
        self.model = xception()
        self.model.to(self.device)
        
        self.optimizer = (optim.Adam(self.model.parameters(), lr=cfg['lr']) if cfg['optimizer'] == 'Adam'
                            else optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg['momentum']))
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=cfg['lr_decay'])

    def criterion(self, pos_scores, neg_scores):
        return F.relu(neg_scores - pos_scores + cfg['hinge_loss_margin']).mean(dim=0)
    
    def train_epoch(self):
        running_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            pos_samples = batch[0].to(device)
            neg_samples = batch[1].to(device)

            pos_scores = self.model(pos_samples)
            neg_scores = self.model(neg_samples)

            loss = self.criterion(pos_scores, neg_scores)
            running_loss += loss.item() / len(self.train_loader)

            loss.backward()
            self.optimizer.step()
        
        return running_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0
        for batch_idx, batch in enumerate(self.val_loader):
            pos_samples = batch[0].to(device)
            neg_samples = batch[1].to(device)

            pos_scores = self.model(pos_samples)
            neg_scores = self.model(neg_samples)

            loss = self.criterion(pos_scores, neg_scores)
            running_loss += loss.item() / len(self.val_loader)
        
        self.model.train()
        
        return running_loss
    
    def train(self):
        for epoch in range(cfg['epochs']):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print("Epoch {}: Train loss = {}, Val loss = {}".format(epoch, train_loss, val_loss))

            self.lr_scheduler.step(val_loss)


if __name__ == '__main__':
    trainer = Trainer()
    # import pdb; pdb.set_trace()
    print(trainer.train_epoch())