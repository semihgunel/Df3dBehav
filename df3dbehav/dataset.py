import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from einops import rearrange
import numpy as np


class DatasetTrain():
    def __init__(
        self, df, n_frames: int, transform=nn.Identity(), background_index:int=7,
    ):
        '''frame: pandas dataframe with path, min and max columns'''
        super().__init__()
        self.df = df
        self.n_frames = n_frames
        self.transform = transform
        self.path_list_unq = np.unique(df['path'].to_numpy())
        self.background_index = background_index
        self.prd = {p: read_pose_result_mm(p) for p in self.path_list_unq}
    
    def get_clip(self, mi:int, ma:int, l:int, n_frames:int):
        curr_len = ma - mi
        missing = n_frames - curr_len
        
        mi = mi - missing // 2
        ma = ma + missing //2 + (missing % 2)
        
        clip = np.arange(mi, ma)
        clip = np.clip(clip, 0, l - 1)
        return clip
    
    def __getitem__(self, idx):
        path, mi, ma, b = self.df.iloc[idx]
        clip = self.get_clip(mi, ma, self.prd[path].shape[0] ,self.n_frames)
        
        target = torch.ones((clip.shape[0],)) * self.background_index
        target[np.logical_and(clip <= ma, clip >= mi)] = b

        pts = self.prd[path][clip]
        pts = self.transform(pts)
        pts = torch.from_numpy(pts) if not torch.is_tensor(pts) else pts
        
        return (
            pts.float(),
            target.long(),
            {"path": path, "min":mi, "max":ma, "behav": b},
        )
    
    def __len__(self):
        return len(self.df)
    
    
    

class DatasetTest():
    def __init__(
        self, pts, n_frames: int, transform=nn.Identity(),
    ):
        super().__init__()        
        
        self.n_frames = n_frames
        self.transform = transform
        
        self.clips = rearrange(pts, '(b t) j c -> b t (j c)', t=self.n_frames)
        
    def __getitem__(self, idx):
        pts = self.clips[idx]
        
        pts = self.transform(pts)
        pts = torch.from_numpy(pts) if not torch.is_tensor(pts) else pts
        mi = idx * self.n_frames
        ma = mi + self.n_frames
        
        return (
            pts.float(),
            {"min":mi, "max":ma},
        )
    
    def __len__(self):
        return self.clips.shape[0]