import cv2
import pandas as pd
import pickle5
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
import sys
import pandas as pd 
from pathlib import Path
import argparse

from .model import ClassifierDf3d
from .dataset import DatasetTest
from .augmentation import TimeSeriesTransformEval
from .utils import read_pose_result 
import pytorch_lightning as pl
import pathlib
from torch.utils.data import DataLoader

torch2np = lambda x : x.cpu().data.numpy()
labels = np.array(['abdominal_grooming', 'antennal_grooming', 'eye_grooming',
                   'foreleg_grooming', 'forward_walking', 'hindleg_grooming',
                   'pushing'])

def main():
    parser = argparse.ArgumentParser(description='Process pose_result files to produce behavior estimates')
    parser.add_argument('--path',  required=True, metavar='P', type=str, help='pose result folder')
    parser.add_argument('-v', '--verbose', action='count', default=0,  help='print debug')
    args = parser.parse_args()
    
    model = ClassifierDf3d.load_from_checkpoint(str(pathlib.Path(__file__).parent.resolve()) + '/../data/epoch=445-step=6243.ckpt')
    pts = read_pose_result(args.path)
    train_dataset = DatasetTest(pts, n_frames=100, transform=TimeSeriesTransformEval())
    loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=False)

    out = np.zeros((pts.shape[0], len(labels)))
    for idx, batch in enumerate(loader):
        x, meta = batch
        y_hat = model(x)
        y_hat = y_hat.softmax(1)
        y_hat = torch2np(y_hat)
        for idx in range(y_hat.shape[0]):
            out[meta["min"][idx]:meta["max"][idx]] = y_hat[idx].T

    print(f"Saving result at {Path(args.path).resolve()}")
    df = pd.DataFrame(out.T, labels).to_csv(args.path + 'behav_clsf.csv')
    
if __name__ == "__main__":
    main()