import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
EXP=Path(__file__).parent
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


from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video
from src.model import DynamicCSNN,CSNN,DynamicResCSNN,ResNetLSTM,ResCSNN,ScalePredictor



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
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--timescale",default=1,type=float,help="何倍に時間をスケールするか. timescale=2でtimewindowが1/2になる.")
    parser.add_argument("--insize",default=128,type=int,help="入力画像のサイズ")
    args = parser.parse_args()

    device=torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    timescale=args.timescale
    time_window=int(3000/timescale)

    savepath=Path(__file__).parent/f"window{time_window}-insize{args.insize}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.insize==128:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
            torch.from_numpy,
        ])
    else:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(args.insize,args.insize)),
            tonic.transforms.ToFrame(sensor_size=(args.insize,args.insize,2),time_window=time_window),
            torch.from_numpy,
        ])


    # cache_path=str(f"/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/cache-data/gesture/window{time_window}-insize{args.insize}")
    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    # testset=tonic.DiskCachedDataset(testset,cache_path=cache_path)
    test_loader = DataLoader(testset,   batch_size=1, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # Validation step
    print_terminal(f"getting sample frames...")
    sequence=300
    test_list={}

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device) #[T x batch x xdim]

            if targets[0].item() in test_list.keys():
                continue

            inputs[inputs>0]=1.0

            if sequence>0 and inputs.shape[0]>sequence*timescale: #configでシーケンスが指定された場合はその長さに切り取る
                inputs=inputs[:int(sequence*timescale)]

            test_list[targets[0].item()]=inputs[:,0] #[T x xdim]
            if len(test_list)>=11:
                break
    print_terminal(f"done\n")


    ##>> 入力確認 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_terminal(f"saving sample videos...")
    for label,inputs in test_list.items():
        frame_np=inputs.to("cpu").detach().numpy()
        frame=1.5*frame_np[:,0]+0.5*frame_np[:,1]-1
        save_heatmap_video(
            frame,
            output_path=savepath,
            file_name=f"gesture_{label}",
            fps=30,scale=int(5),
            frame_label_view=False,
        )

    print_terminal("done")

if __name__=="__main__":
    main()