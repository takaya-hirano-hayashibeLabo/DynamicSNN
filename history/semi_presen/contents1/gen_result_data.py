from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))
# print(ROOT)

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
from src.model import DynamicSNN
from src.utils import load_yaml,print_terminal


def plot_and_save_inputs(base_input, scaled_input, save_path, batch_index=0,window=10):
    plt.figure(figsize=(12, 6))
    
    # Base Inputの描画
    plt.subplot(2, 1, 1)
    plt.title("Base Input")
    plt.plot(base_input[:,batch_index].flatten().cpu().numpy(),"o")
    plt.ylim(0.5,1.5)
    plt.xticks(ticks=np.arange(0, base_input.shape[0], window)-0.5)  # x軸の間隔をwindowに設定
    plt.grid()
    # print(base_input[:,batch_index].flatten().cpu().numpy())
    
    # Scaled Inputの描画
    plt.subplot(2, 1, 2)
    plt.title("Scaled Input")
    plt.plot(scaled_input[:,batch_index].flatten().cpu().numpy(),"o")
    plt.ylim(0.5,1.5)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--testnums",type=int,default=10)
    parser.add_argument("--timescale",type=float,default=1.0)
    parser.add_argument("--device",default=0)
    args=parser.parse_args()

    device=torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    config=load_yaml(ROOT/"history/semi_presen/contents1/dynasnn_conf.yml") #DynaSNNの設定
    print_terminal(f"running processing [timescale: {args.timescale:.2f}]...")

    
    result_trajectory=[]
    for i_test in tqdm(range(args.testnums)):
        model=DynamicSNN(conf=config["model"]).to(device) # テストエポックごとにモデルを初期化する. なぜなら未学習のモデルの出力は初期重みに死ぬほど依存するから

        T=300
        batch=1000 #バッチよりもモデルの初期値に依存する
        insize=config["model"]["in-size"]

        p=0.1
        base_input=torch.where(
            torch.rand(size=(T,batch,insize))<p,1.0,0.0
        ).to(device)
        base_s,base_i,base_v=model(base_input)

        a=args.timescale  # 'a' can now be a float
        # Create scaled_input by shifting indices by a factor of 'a'
        scaled_input = torch.zeros(size=(int(a * T), batch, insize)).to(device)
        if a >= 1.0:
            kernel_size=a
            for t in range(T):
                scaled_index = int(a * t)
                if scaled_index < scaled_input.shape[0]:
                    scaled_input[scaled_index] = base_input[t]
        else:
            # 畳み込みを行う処理
            kernel_size = math.ceil(1 / a)  # カーネルサイズを設定
            
            scaled_input = F.conv1d(base_input.permute(1, 2, 0), 
                                            weight=torch.ones(1, 1, kernel_size).to(base_input.device), 
                                            stride=kernel_size).permute(2, 0, 1)
            scaled_input[scaled_input<0.5]=0.0
            scaled_input[scaled_input>=0.5]=1.0

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
    result_path=ROOT/f"history/semi_presen/contents1/results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path/f"result_timescale{args.timescale:.2f}.json",'w') as f:
        json.dump(result_dict,f,indent=4)

    inspike_img_path=ROOT/f"history/semi_presen/contents1/imgs/inspike/"
    if not os.path.exists(inspike_img_path):
        os.makedirs(inspike_img_path)
    plot_and_save_inputs(base_input,scaled_input,inspike_img_path/f"inspikes_timescale{args.timescale:.2f}.png",batch_index=0,window=kernel_size)
    # print(base_input)

if __name__ == "__main__":
    main()