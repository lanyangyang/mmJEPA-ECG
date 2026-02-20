import os
from mimetypes import inited
from pathlib import Path
import math
import numpy as np
import torch
from numpy.lib._stride_tricks_impl import sliding_window_view
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

class npyLoad(Dataset):
    def __init__(self, npy_path, device='cpu'):

        data = np.load(npy_path, allow_pickle=True).item()
        self.rcg = torch.tensor(data['rcg'], dtype=torch.float32)
        self.ecg = torch.tensor(data['ecg'], dtype=torch.float32)
        self.pos = torch.tensor(data['pos'], dtype=torch.float32)
        self.y   = torch.tensor(data['y'],   dtype=torch.long)

        if device != 'cpu':
            self.rcg = self.rcg.to(device)
            self.ecg = self.ecg.to(device)
            self.pos = self.pos.to(device)
            self.y   = self.y.to(device)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        rcg = self.rcg[idx]
        ecg = self.ecg[idx]
        pos = self.pos[idx]

        return rcg, ecg, pos, self.y[idx]

class npyLoadWithHR(Dataset):
    def __init__(self, npy_path, device='cpu'):

        data = np.load(npy_path, allow_pickle=True).item()
        self.rcg = torch.tensor(data['rcg'], dtype=torch.float32)
        self.ecg = torch.tensor(data['ecg'], dtype=torch.float32)
        self.pos = torch.tensor(data['pos'], dtype=torch.float32)
        self.y   = torch.tensor(data['y'],   dtype=torch.long)
        self.hr  = torch.tensor(data['hr'],  dtype=torch.float32)

        if device != 'cpu':
            self.rcg = self.rcg.to(device)
            self.ecg = self.ecg.to(device)
            self.pos = self.pos.to(device)
            self.y   = self.y.to(device)
            self.hr  = self.hr.to(device)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        rcg = self.rcg[idx]
        ecg = self.ecg[idx]
        pos = self.pos[idx]

        return rcg, ecg, pos, self.y[idx], self.hr[idx]
