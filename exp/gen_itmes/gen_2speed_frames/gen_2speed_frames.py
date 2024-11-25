import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
EXP=Path(__file__).parent
import sys
sys.path.append(str(ROOT))
print(str(ROOT))

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
    parser.add_argument("--saveto",required=True,help="結果を保存するディレクトリ")
    parser.add_argument("--timescale1",type=float,default=1,help="前半の速度")
    parser.add_argument("--timescale2",type=float,default=1,help="後半の速度")
    parser.add_argument("--sequence",type=int,default=300,help="時系列のタイムシーケンス")
    parser.add_argument("--timewindow",type=int,default=3000,help="フレームの時間ウィンドウ")
    parser.add_argument("--insize",type=int,default=128,help="フレームのサイズ")
    args = parser.parse_args()


    timescale1,timescale2=args.timescale1,args.timescale2
    savepath=Path(args.saveto)/f"timescale{timescale1}-{timescale2}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    minibatch=8
    sequence=args.sequence #時系列のタイムシーケンス
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=int(args.timewindow/timescale1)
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


    testset1=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    # cache_path=str(EXP/f"test-cache/gesture-window{time_window}")
    # testset1=tonic.DiskCachedDataset(testset1,cache_path=cache_path)
    test_loader1 = DataLoader(testset1,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)



    time_window=int(args.timewindow/timescale2)
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


    # cache_path=str(EXP/f"test-cache/gesture-window{time_window}")
    # testset2=tonic.DiskCachedDataset(testset2,cache_path=cache_path)
    testset2=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    test_loader2 = DataLoader(testset2,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # Validation step
    print_terminal(f"timescale: ({timescale1}, {timescale2})"+"-"*500)

    with torch.no_grad():
        for (inputs1, targets1), (inputs2, targets2) in zip(tqdm(test_loader1), tqdm(test_loader2)):

            seq1,seq2=int(sequence/2),int(sequence/2)
            inputs1=inputs1[:,:int(seq1*timescale1)]
            inputs2=inputs2[:,int(seq1*timescale2):int(seq1*timescale2+seq2*timescale2)]

            inputs1, targets1 = inputs1.to(device).permute((1, 0, *[i + 2 for i in range(inputs1.ndim - 2)])).to(torch.float), targets1.to(device)
            inputs2, targets2 = inputs2.to(device).permute((1, 0, *[i + 2 for i in range(inputs2.ndim - 2)])).to(torch.float), targets2.to(device)
            
            
            inputs=torch.cat(
                [inputs1,inputs2], dim=0
            )
            inputs[inputs>0]=1.0

            break #1バッチとれればいい

    print_terminal(f"done\n")


    ##>> 入力確認 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_terminal(f"saveing sample videos...")
    video_size=320
    for i_frame in tqdm(range(minibatch)):
        frame_np=inputs[:,i_frame].to("cpu").detach().numpy()
        frame=1.5*frame_np[:,0]+0.5*frame_np[:,1]-1
        save_heatmap_video(
            frame,
            output_path=savepath/"video",
            file_name=f"train_input_label{targets1[i_frame]}",
            fps=60,scale=int(video_size/insize),
            frame_label_view=False
        )
    print_terminal("done")
    ##<< 入力確認 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    save_dict2json(
        vars(args),saveto=savepath/"args.json"
    )

    result={
        "timescale1":args.timescale1,
        "timescale2":args.timescale2,
    }
    save_dict2json(
        result,saveto=savepath/"result.json"
    )


if __name__=="__main__":
    main()