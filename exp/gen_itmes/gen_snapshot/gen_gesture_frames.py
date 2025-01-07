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


from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap, resize_heatmap, apply_cmap2heatmap


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # numpy.ndarrayをTensorに変換
    inputs = [input.clone().detach() for input in inputs]
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor


def save_snapshot(input_list:list[np.ndarray],cmap:str,scale:int,savedir:Path,filename:str):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    num_frames=len(input_list)
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames*2, 2))  # Use tight_layout
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)  # Reduce wspace for tighter spacing
    for i, ax in enumerate(axes):
        frame = resize_heatmap(input_list[i], scale)
        frame = apply_cmap2heatmap(frame, cmap)
        ax.imshow(frame)
        ax.axis("off")
        
    fig.savefig(savedir/filename)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--timewindow",default=3000,type=int)
    parser.add_argument("--insize",default=128,type=int)
    parser.add_argument("--framenum",default=3,type=int)
    parser.add_argument("--cmap",default="plasma",type=str)
    parser.add_argument("--scale",default=1,type=int)
    args = parser.parse_args()

    timewindow=args.timewindow
    insize=args.insize
    framenum=args.framenum

    savedir=Path(__file__).parent/f"timewindow{timewindow}"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

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

            skip_frames=int(inputs.shape[1]//framenum)
            input_heatmaps=[]
            for t, frame_i in enumerate(inputs[0]):
                if len(input_heatmaps)>=framenum:
                    break

                if t%skip_frames==0:
                    frame_np=frame_i.to("cpu").detach().numpy()
                    frame_np=1.5*frame_np[0]+0.5*frame_np[1]-1
                    input_heatmaps.append(frame_np)
            save_snapshot(
                input_heatmaps,cmap=args.cmap,scale=args.scale,
                savedir=savedir/f"svgs/gesture_{targets[0].item()}",
                filename=f"label{targets[0].item()}.svg"
            )

            saved_labels.append(targets[0].item())

            print(f"[{len(saved_labels)}/{len(tonic.datasets.DVSGesture.classes)}] saved label:{saved_labels[-1]}")

            if len(saved_labels)==len(tonic.datasets.DVSGesture.classes):
                break

    save_dict2json(
        vars(args),saveto=savedir/"args.json"
    )

    print_terminal("done")


if __name__=="__main__":
    main()