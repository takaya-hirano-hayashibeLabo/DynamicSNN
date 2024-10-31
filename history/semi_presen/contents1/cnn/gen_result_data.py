from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))
print(ROOT)

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import argparse
import numpy as np
import json
import os
import math
from tqdm import tqdm
from src.model import DynamicCSNN,DynamicResCSNN
from src.utils import load_yaml,print_terminal


def plot_and_save_inputs(base_input:torch.Tensor, scaled_input:torch.Tensor, save_path:Path, batch_index:int=0,window:int=10):

    base_in_plot=base_input[:,batch_index].flatten(start_dim=1).cpu().numpy()
    scaled_in_plot=scaled_input[:,batch_index].flatten(start_dim=1).cpu().numpy()

    xdim=base_input.shape[-1]

    plt.figure(figsize=(12, 6))
    
    # Base Inputの描画
    plt.subplot(2, 1, 1)
    plt.title("Base Input")
    for i in range(xdim):
        plt.plot(base_in_plot[:,i]*(i+1),"o")
    plt.ylim(0.5,xdim+0.5)
    plt.xticks(ticks=np.arange(0, base_input.shape[0], window)-0.5)  # x軸の間隔をwindowに設定
    plt.grid()
    # print(base_input[:,batch_index].flatten().cpu().numpy())
    
    # Scaled Inputの描画
    plt.subplot(2, 1, 2)
    plt.title("Scaled Input")
    for i in range(xdim):
        plt.plot(scaled_in_plot[:,i]*(i+1),"o")
    plt.ylim(0.5,xdim+0.5)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--testnums",type=int,default=10)
    parser.add_argument("--timescale",type=float,default=1.0)
    parser.add_argument("--device",default=0)
    parser.add_argument("--saveto",default="")
    parser.add_argument("--confpath",default="")
    args=parser.parse_args()

    device=torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    config=load_yaml(args.confpath) #DynaSNNの設定
    print_terminal(f"running processing [timescale: {args.timescale:.2f}]...")

    
    result_trajectory=[]
    for i_test in tqdm(range(args.testnums)):

        if "cnn-type" in config["model"] and config["model"]["cnn-type"]=="res":
            model=DynamicResCSNN(conf=config["model"]).to(device) # テストエポックごとにモデルを初期化する. なぜなら未学習のモデルの出力は初期重みに死ぬほど依存するから
        else:
            model=DynamicCSNN(conf=config["model"]).to(device) # テストエポックごとにモデルを初期化する. なぜなら未学習のモデルの出力は初期重みに死ぬほど依存するから

        T=300
        batch=500 #バッチよりもモデルの初期値に依存する
        insize=config["model"]["in-size"]
        in_channel=config["model"]["in-channel"]

        p=0.1
        base_input=torch.where(
            torch.rand(size=(T, batch, in_channel,insize,insize))<p,1.0,0.0
        ).to(device)
        base_s,base_i,base_v=model(base_input)

        a=args.timescale  # 'a' can now be a float
        # Create scaled_input by shifting indices by a factor of 'a'
        scaled_input = torch.zeros(size=(int(a * T), batch, in_channel,insize,insize)).to(device)
        if a >= 1.0:
            kernel_size=a
            for t in range(T):
                scaled_index = int(a * t)
                if scaled_index < scaled_input.shape[0]:
                    scaled_input[scaled_index] = base_input[t]
        else:
            # 1次元の畳み込みを時間方向に行う処理
            kernel_size = math.ceil(1 / a)  # カーネルサイズを設定
            
            # Permute base_input to bring the time dimension to the last position
            base_input_1d = base_input.permute(1, 2, 3, 4, 0)  # (batch, channel, w, h, T)
            
            # Reshape for 1D convolution
            reshaped_input = base_input_1d.view(batch * in_channel * insize * insize, T)
            
            # Create a weight tensor with the same number of input channels
            weight = torch.ones(1, 1, kernel_size).to(base_input.device)
            
            # Apply 1D convolution along the time dimension
            scaled_input = F.conv1d(reshaped_input.unsqueeze(1), 
                                    weight=weight, 
                                    stride=kernel_size).view(batch, in_channel, insize, insize, -1).permute(4, 0, 1, 2, 3)
            
            scaled_input[scaled_input<0.5]=0.0
            scaled_input[scaled_input>=0.5]=1.0

            # print(scaled_input.shape)

        org_s,org_i,org_v=model.forward(scaled_input)
        scaled_s,scaled_i,scaled_v=model.dynamic_forward_v1(scaled_input,a=torch.Tensor([a for _ in range(scaled_input.shape[0])]))
        
        
        v1_resampled=F.interpolate(base_v.permute(1,2,0), size=int(a*T), mode='linear', align_corners=False).permute(-1,0,1) #基準膜電位のタイムスケール(線形補間)
        
        scaled_T=scaled_input.shape[0]
        mse_lif=  np.mean((v1_resampled.to("cpu").detach().numpy()[:scaled_T]-org_v.to("cpu").detach().numpy())**2) 
        mes_dyna= np.mean((v1_resampled.to("cpu").detach().numpy()[:scaled_T]-scaled_v.to("cpu").detach().numpy())**2) 

        result_trajectory.append((mse_lif,mes_dyna))

    result_trajectory=np.array(result_trajectory)
    result_dict={
        "timescale":args.timescale,
        "testnums":args.testnums,
        "lif_mean":np.mean(result_trajectory[:,0]).astype(float),
        "lif_std":np.std(result_trajectory[:,0]).astype(float),
        "dyna_mean":np.mean(result_trajectory[:,1]).astype(float),
        "dyna_std":np.std(result_trajectory[:,1]).astype(float),
    }

    result_path=Path(__file__).parent/args.saveto
    if not os.path.exists(result_path/"json"):
        os.makedirs(result_path/"json")

    with open(result_path/f"json/result_timescale{args.timescale:.2f}.json",'w') as f:
        json.dump(result_dict,f,indent=4)

    inspike_img_path=result_path/"imgs/inspike/"
    if not os.path.exists(inspike_img_path):
        os.makedirs(inspike_img_path)
    plot_and_save_inputs(base_input,scaled_input,inspike_img_path/f"inspikes_timescale{args.timescale:.2f}.png",batch_index=0,window=kernel_size)
    # print(base_input)

if __name__ == "__main__":
    main()