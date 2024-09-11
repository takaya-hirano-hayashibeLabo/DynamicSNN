import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from snntorch import functional as SF
from torch.utils.data import DataLoader
import pandas as pd
import torchvision
import tonic
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from math import floor


from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # numpy.ndarrayをTensorに変換
    inputs = [input.clone().detach() for input in inputs]
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--timescale",default=1,type=int,help="何倍に時間をスケールするか. timescale=2でtimewindowが1/2になる.")
    parser.add_argument("--timewindow",default=3000,type=int)
    parser.add_argument("--insize",default=128,type=int)
    parser.add_argument("--sequence",default=300,type=int)
    parser.add_argument("--framenum",default=5,type=int)
    args = parser.parse_args()

    timescale=args.timescale
    timewindow=args.timewindow
    insize=args.insize
    sequence=args.sequence
    framenum=args.framenum

    print(tonic.datasets.DVSGesture.classes, len(tonic.datasets.DVSGesture.classes))

    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if insize==128:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=timewindow),
            torch.from_numpy,
        ])
    else:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
            tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=timewindow),
            torch.from_numpy,
        ])


    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    test_loader = DataLoader(testset,   batch_size=1, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print_terminal(f"Capturing frames...")
    saved_labels=[]
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader): #inputs:[N x T x xdim]

            if targets[0].item() in saved_labels: #同じラベルはもういらない
                continue

            inputs[inputs>0]=1.0
            if sequence>0 and inputs.shape[0]>sequence*timescale: #configでシーケンスが指定された場合はその長さに切り取る
                inputs=inputs[:,:int(sequence*timescale)]

            skip_frames=int(sequence*timescale//framenum)
            for t, frame_i in enumerate(inputs[0]):
                if t>int(sequence*timescale):
                    break
                if t%skip_frames==0:
                    frame_np=frame_i.to("cpu").detach().numpy()
                    frame_np=1.5*frame_np[0]+0.5*frame_np[1]-1
                    save_heatmap(frame_np,Path(__file__).parent/f"imgs/gesture_{targets[0].item()}",f"label{targets[0].item()}_frame{t}.svg")

            saved_labels.append(targets[0].item())

            print(f"[{len(saved_labels)}/{len(tonic.datasets.DVSGesture.classes)}] saved label:{saved_labels[-1]}")

            if len(saved_labels)==len(tonic.datasets.DVSGesture.classes):
                break

    save_dict2json(
        vars(args),saveto=Path(__file__).parent/"args.json"
    )

    print_terminal("done")


if __name__=="__main__":
    main()