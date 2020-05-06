import torch
from scipy.io import loadmat
from data.filter import get_filtered_indices
from archs.xception import xception
from archs.simple_cnn import simple_cnn
from config.config import cfg
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from data.SUNRGBD import SUNRGBD
import numpy as np
from torch.utils.data import DataLoader


def get_roc_score(loader, device, model):
    model.eval()
    pos_predictions = []
    neg_predictions = []
    for batch_idx, batch in enumerate(loader):
        pos_samples = batch[0].to(device).float()
        neg_samples = batch[1].to(device).float()

        # Pass the positive and negative samples together in a single batch
        B = pos_samples.shape[0]
        all_samples = torch.cat([pos_samples, neg_samples], dim=0)
        all_scores = model(all_samples)
        # Separate the positive and negative scores
        pos_scores = all_scores[:B]
        neg_scores = all_scores[B:]

        pos_predictions.append(pos_scores.cpu().detach().numpy())
        neg_predictions.append(neg_scores.cpu().detach().numpy())
    
    pos_predictions = (np.concatenate(pos_predictions)[:, 0] + 1) / 2
    neg_predictions = (np.concatenate(neg_predictions)[:, 0] + 1) / 2
    pos_labels = np.ones(pos_predictions.shape, dtype=int)
    neg_labels = np.zeros(neg_predictions.shape, dtype=int)

    predictions = np.concatenate([pos_predictions, neg_predictions])
    labels = np.concatenate([pos_labels, neg_labels])

    roc_score = roc_auc_score(labels, predictions)

    return roc_score


if __name__ == "__main__":
    device = torch.device('cuda' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    model = simple_cnn(num_objects=len(cfg['CLASSES']))
    model.load_state_dict(torch.load(cfg["best_model_path"])["params"])
    model.to(device)
    
    data = loadmat(cfg['data_path'])['SUNRGBDMeta'].squeeze()
    filtered_indices = get_filtered_indices(data)
    idx_trainval, idx_test = train_test_split(filtered_indices, test_size=0.1, random_state=1)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.3, random_state=1)
    # import pdb; pdb.set_trace()
    train_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_train], split='train')
    val_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_val], split='val')
    test_dataset = SUNRGBD(cfg['data_root'], cfg['cache_dir'], data[idx_test], split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=cfg['num_workers'])

    val_roc_score = get_roc_score(val_loader, device, model)
    # train_roc_score = get_roc_score(train_loader, device, model)

    # import pdb; pdb.set_trace()
    print(val_roc_score)
    