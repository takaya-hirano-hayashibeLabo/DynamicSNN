import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(ROOT))

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from snntorch import functional as SF
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torchvision
import tonic
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


from src.utils import print_terminal


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # numpy.ndarrayをTensorに変換
    inputs = [torch.tensor(input) for input in inputs]
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timewindow",default=3000,type=int)
    parser.add_argument("--insize",default=32,type=int)
    args = parser.parse_args()



    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=args.timewindow
    insize=args.insize
    if insize==128:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
            torch.from_numpy,
        ])
    else:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
            tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
            torch.from_numpy,
        ])


    trainset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=True,transform=transform)
    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)

    cachepath=ROOT/f"cache-data/gesture/window{time_window}-insize{insize}"
    trainset=tonic.DiskCachedDataset(trainset,cache_path=str(cachepath/"train"))
    testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True,collate_fn=custom_collate_fn ,num_workers=3)
    test_loader = DataLoader(testset,   batch_size=64, shuffle=False,collate_fn=custom_collate_fn,num_workers=3)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    print_terminal(f"generating cache: datatype@gesture window@{time_window} insize@{insize}"+"="*1000)
    print("[train set]")
    for _ in tqdm(train_loader):
        pass
    print("[test set]")
    for _ in tqdm(test_loader):
        pass
    print_terminal("done.")


if __name__=="__main__":
    main()