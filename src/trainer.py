import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from archs.xception import xception
from archs.simple_cnn import simple_cnn
from data.filter import get_filtered_indices
from data.SUNRGBD import SUNRGBD
from config.config import cfg
from src.utils import AvgMeter

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self):
        data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
        filtered_indices = get_filtered_indices(data)
        idx_trainval, idx_test = train_test_split(filtered_indices, test_size=0.1, random_state=1)
        idx_train, idx_val = train_test_split(idx_trainval, test_size=0.3, random_state=1)
        train_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_train], split='train')
        val_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_val], split='val')
        
        print("Number of training instances:", len(train_dataset))
        print("Number of validation instances:", len(val_dataset))

        self.train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                        num_workers=cfg['num_workers'])
        self.val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                                        num_workers=cfg['num_workers'])

        self.device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')

        if cfg['model_type'] == 'xception':
            self.model = xception(num_objects=len(cfg['CLASSES']))
        else:
            self.model = simple_cnn(num_objects=len(cfg['CLASSES']))
        self.model.to(self.device)
        
        self.optimizer = (optim.Adam(self.model.parameters(), lr=cfg['lr']) if cfg['optimizer'] == 'Adam'
                            else optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg['momentum']))
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=cfg['lr_decay'])

        self.logger = SummaryWriter()

        self.best_save_path = cfg['best_model_path']
        self.latest_save_path = cfg['latest_model_path']

    def save(self, epoch, batch_index, val_loss):
        ckpt = {
            'params': self.model.state_dict(),
            'optim' : self.optimizer.state_dict(),
            'epoch' : epoch,
            'batch' : batch_index,
            'val_loss': val_loss
        }
        torch.save(ckpt, self.latest_save_path)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, self.best_save_path)

    def criterion(self, pos_scores, neg_scores):
        return F.relu(neg_scores - pos_scores + cfg['hinge_loss_margin']).mean(dim=0)
    
    def train_epoch(self, epoch):
        running_loss = AvgMeter()
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            pos_samples = batch[0].to(self.device).float()
            neg_samples = batch[1].to(self.device).float()

            # Pass the positive and negative samples together in a single batch
            B = pos_samples.shape[0]
            all_samples = torch.cat([pos_samples, neg_samples], dim=0)
            all_scores = self.model(all_samples)
            # Separate the positive and negative scores
            pos_scores = all_scores[:B]
            neg_scores = all_scores[B:]

            loss = self.criterion(pos_scores, neg_scores)
            running_loss.update(loss.item()*pos_scores.shape[0], pos_scores.shape[0])

            iteration = epoch * len(self.train_loader) + batch_idx
            self.logger.add_scalars('Loss', {'train':loss.item()}, iteration)

            if iteration % cfg['log_every'] == 0:
                print("Epoch {}, Batch {}, Iteration {}: Traning loss = {}".format(epoch, batch_idx,
                        iteration, loss.item()))
            
            if iteration % cfg['val_every'] == 0:
                val_loss = self.validate()
                print("Epoch {}, Batch {}, Iteration {}: Validation loss = {}".format(epoch, batch_idx,
                        iteration, val_loss))
                self.logger.add_scalars('Loss', {'val':val_loss}, iteration)
                self.lr_scheduler.step(val_loss)
                self.save(epoch, batch_idx, val_loss)

            loss.backward()
            self.optimizer.step()
        
        return running_loss.get_avg()
    
    @torch.no_grad()
    def validate(self):
        running_loss = AvgMeter()
        self.model.eval()
        for batch_idx, batch in enumerate(self.val_loader):
            pos_samples = batch[0].to(self.device).float()
            neg_samples = batch[1].to(self.device).float()

            # Pass the positive and negative samples together in a single batch
            B = pos_samples.shape[0]
            all_samples = torch.cat([pos_samples, neg_samples], dim=0)
            all_scores = self.model(all_samples)
            # Separate the positive and negative scores
            pos_scores = all_scores[:B]
            neg_scores = all_scores[B:]

            loss = self.criterion(pos_scores, neg_scores)
            running_loss.update(loss.item()*pos_samples.shape[0], pos_samples.shape[0])
        
        self.model.train()
        
        return running_loss.get_avg()
    
    def train(self):
        self.best_val_loss = np.inf
        for epoch in range(cfg['epochs']):
            train_loss = self.train_epoch(epoch)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    # import pdb; pdb.set_trace()