import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
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
import torch.nn.functional as F
from matplotlib import rcParams
import seaborn as sns
sns.set(style="darkgrid")


from src.utils import load_yaml,print_terminal,save_dict2json
from src.model import DynamicCSNN,CSNN,DynamicResCSNN,ResNetLSTM,ResCSNN,ScalePredictor



def plot_firingrate_element(out_s_base, out_s_scaled, n, window, saveto):
    """
    Plot the n-th batch element of out_s_ideal and out_s_scaled.
    
    :param out_s_ideal: [N x T x xdim]
    :param out_s_scaled: [N x T x xdim]
    :param n: index of the batch element to plot
    :param window: window size for calculating firing rate
    """
    rcParams['font.family'] = 'serif'
    xdim = out_s_base.shape[2]
    
    def calculate_firing_rate(spikes, w):
        firing_rate = np.zeros_like(spikes, dtype=float)
        for t in range(w, spikes.shape[0]):
            firing_rate[t] = np.mean(spikes[t-w:t])
        firing_rate[:w] = None  # Set initial values to None
        return firing_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    cmap = plt.get_cmap('viridis')
    
    scale=int(out_s_scaled.shape[1]/out_s_base.shape[1])
    for i in range(xdim):
        firing_rate_ideal = calculate_firing_rate(out_s_base[n, :, i].cpu().numpy(), window)
        firing_rate_scaled = calculate_firing_rate(out_s_scaled[n, :, i].cpu().numpy(), window*scale)
        
        T_base,T_scaled=out_s_base.shape[1],out_s_scaled.shape[1]
        ax1.plot(np.array(range(T_base))/T_base, firing_rate_ideal, label=f'xdim {i}',      color=cmap(i/xdim), linewidth=3)
        ax2.plot(np.array(range(T_scaled))/T_scaled, firing_rate_scaled, label=f'xdim {i}', color=cmap(i/xdim), linewidth=3)
    
    ax1.set_title(f'Output spike density $a=1.0$ / SNN', fontsize=26)
    ax1.set_ylabel('Density', fontsize=26)
    ax1.set_xlim(np.min((np.array(range(T_base))[window::])/T_base),1)
    ax1.tick_params(axis='y', labelsize=20)  # Set fontsize for y-ticks
    ax1.tick_params(axis='x', labelbottom=False) 

    ax2.set_title(f'Output spike density $a={scale:.1f}$ / SNN', fontsize=26)
    ax2.set_xlabel('Time', fontsize=26)
    ax2.set_ylabel('Density', fontsize=26)
    ax2.set_xlim(np.min((np.array(range(T_base))[window::])/T_base),1)
    ax2.tick_params(axis='x', labelsize=20)  # Set fontsize for x-ticks
    ax2.tick_params(axis='y', labelsize=20)  # Set fontsize for y-ticks

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i/xdim), 
        edgecolor='none', label=f'#{i+1}')
        for i in range(xdim)
    ]
    legend = ax1.legend(
            handles=legend_handles, bbox_to_anchor=(0.5, 1.8), loc='upper center',
            frameon=False, ncol=6, fontsize=18,
        )
    ax1.add_artist(legend)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Adjust the top margin to make space for the legend
    plt.savefig(saveto)



def plot_spike_element(out_s_ideal, out_s_scaled, n, saveto):
    """
    Plot the n-th batch element of out_s_ideal and out_s_scaled.
    
    :param out_s_ideal: [N x T x xdim]
    :param out_s_scaled: [N x T x xdim]
    :param n: index of the batch element to plot
    """
    T = out_s_ideal.shape[1]
    xdim = out_s_ideal.shape[2]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for i in range(xdim):
        ax1.scatter(range(T), out_s_ideal[n, :, i].cpu().numpy() * (i+1), label=f'xdim {i}', s=1)
        ax2.scatter(range(T), out_s_scaled[n, :, i].cpu().numpy() * (i+1), label=f'xdim {i}', s=1)
    
    ax1.set_title(f'out_s_ideal - Batch element {n}')
    ax1.set_xlabel('T')
    ax1.set_ylabel('Spike Value')
    ax1.set_ylim(0, xdim+1)
    ax1.legend()
    
    ax2.set_title(f'out_s_scaled - Batch element {n}')
    ax2.set_xlabel('T')
    ax2.set_ylabel('Spike Value')
    ax2.set_ylim(0, xdim+1)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(saveto)


def plot_batch_element(out_v_ideal, out_v_scaled, n, saveto, head_t=50):
    """
    Plot the n-th batch element of out_v_ideal and out_v_scaled.
    
    :param out_v_ideal: [N x T x xdim]
    :param out_v_scaled: [N x T x xdim]
    :param n: index of the batch element to plot
    """
    T = out_v_ideal.shape[1]
    xdim = out_v_ideal.shape[2]

    # # Normalize to range [-1, 1]
    # out_v_ideal = 2 * (out_v_ideal - out_v_ideal[n].min()) / (out_v_ideal[n].max() - out_v_ideal[n].min()) - 1
    # out_v_scaled = 2 * (out_v_scaled - out_v_scaled[n].min()) / (out_v_scaled[n].max() - out_v_scaled[n].min()) - 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for i in range(xdim):
        ax1.plot(np.array(range(T))[head_t:], out_v_ideal[n, head_t:, i].cpu().numpy(), label=f'xdim {i}')
        ax2.plot(np.array(range(T))[head_t:], out_v_scaled[n, head_t:, i].cpu().numpy(), label=f'xdim {i}')
    
    ax1.set_title(f'out_v_ideal - Batch element {n}')
    ax1.set_xlabel('T')
    ax1.set_ylabel('Value')
    ax1.legend()
    
    ax2.set_title(f'out_v_scaled - Batch element {n}')
    ax2.set_xlabel('T')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(saveto)

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
    parser.add_argument('--target', type=str,help="configのあるパス")
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--timescale",default=1,type=int,help="何倍に時間をスケールするか. timescale=2でtimewindowが1/2になる.")
    parser.add_argument("--saveto",required=True,help="結果を保存するディレクトリ")
    parser.add_argument("--modelname",default="model_best.pth",help="モデルのファイル名")
    parser.add_argument("--testnum",type=int,default=5,help="stdを求めるために何回testするか")
    parser.add_argument("--droprate",type=float, default=0.3, help="テスト時にdropする割合")
    args = parser.parse_args()

    timescale=args.timescale
    testnum=args.testnum

    if not os.path.exists(Path(args.saveto) ):
        os.makedirs(Path(args.saveto))


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf:dict
    train_conf,model_conf=conf["train"],conf["model"]

    # minibatch=train_conf["batch"]
    minibatch=32
    sequence=train_conf["sequence"] #時系列のタイムシーケンス
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=DynamicResCSNN(model_conf)
        else:
            model=DynamicCSNN(model_conf)
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=ResCSNN(model_conf)
        else:
            model=CSNN(model_conf)
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="lstm".casefold():
        model=ResNetLSTM(model_conf)
        criterion=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    modelname=args.modelname if ".pth" in args.modelname else args.modelname+".pth"
    model.load_state_dict(torch.load(Path(args.target)/f"result/{modelname}",map_location=device))
    model.to(device)
    model.output_mem=True #membrane potentialを出力するようにする
    model.eval()
    scale_predictor=ScalePredictor(datatype="gesture")
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> 基準データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=int(train_conf["timewindow"]) if "timewindow" in train_conf.keys() else int(train_conf["time-window"])
    if model_conf["in-size"]==128:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
            torch.from_numpy,
        ])
    else:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(model_conf["in-size"],model_conf["in-size"])),
            tonic.transforms.ToFrame(sensor_size=(model_conf["in-size"],model_conf["in-size"],2),time_window=time_window),
            torch.from_numpy,
        ])


    cache_path=str(f"/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/exp/exp_gesture_1/test-cache/gesture-window{time_window}")
    # cache_path=str(EXP/f"test-cache/gesture-window{time_window}")
    testset_base=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    testset_base=tonic.DiskCachedDataset(testset_base,cache_path=cache_path)
    test_loader_base = DataLoader(testset_base,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)
    #<< 基準データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> 基準データを通す >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    out_s_base=[] # [testnum x N x T x xdim]
    out_v_base=[] # [testnum x N x T x xdim]
    with torch.no_grad():
        print_terminal(f"Forwarding base scale input"+"-"*500)
        for inputs, targets in tqdm(test_loader_base):

            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
            inputs[inputs>0]=1.0

            if sequence>0 and inputs.shape[0]>sequence: #configでシーケンスが指定された場合はその長さに切り取る
                inputs=inputs[:int(sequence)]

            if not model_conf["type"]=="dynamic-snn":
                out_s, out_i, out_v = model(inputs) #[T x N x xdim]
            else:
                out_s, out_i, out_v = model.dynamic_forward_v1(
                    s=inputs,a=torch.ones(inputs.shape[0])
                    )
                # out_s, out_i, out_v = model.dynamic_forward(
                #     s=inputs,scale_predictor=scale_predictor
                # )
            out_s_base.append(out_s.permute(1,0,-1)) # [N x T x xdim]
            out_v_base.append(out_v.permute(1,0,-1)) # [N x T x xdim]

        # print(f"out_s_base length: {len(out_s_base)}, {out_s_base[0].shape}")
        out_s_base=torch.cat(out_s_base,dim=0) # [N x T x xdim]
        out_v_base=torch.cat(out_v_base,dim=0) # [N x T x xdim]
    print(f"out_s_base: {out_s_base.shape}")
    #<< 基準データを通す <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    def scale_v(x,scale)->torch.Tensor:
        """
        :param x: [N x T x xdim]
        :return scaled_x: [N x T*scale x xdim]
        """

        # interpolateは引き伸ばしたい軸を一番最後に持ってくる
        scaled_x=F.interpolate(x.permute(0,2,1), size=int(x.shape[1]*scale), mode='linear', align_corners=False).permute(0,2,1)
        return scaled_x

    def scale_s(x,scale)->torch.Tensor:
        """
        :param x: [N x T x xdim]
        :return scaled_x: [N x T*scale x xdim]
        """
        T=x.shape[1]
        scaled_x=torch.zeros(size=(x.shape[0],int(T*scale),x.shape[2]))
        for t in range(T):
            scaled_x[:,int(t*scale),:]=x[:,t,:]
        return scaled_x.to(x.device)

    #>> 理想的なスケーリングを計算 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    out_v_ideal=scale_v(out_v_base,timescale)
    out_s_ideal=scale_s(out_s_base,timescale) #spikeは線形補間だとおかしい
    #<< 理想的なスケーリングを計算 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    print("ideal shape, base shape")
    print(out_v_ideal.shape, out_v_base.shape)


    #>> scalingしたデータの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=int(train_conf["timewindow"]/timescale) if "timewindow" in train_conf.keys() else int(train_conf["time-window"]/timescale)
    if model_conf["in-size"]==128:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
            torch.from_numpy,
        ])
    else:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(model_conf["in-size"],model_conf["in-size"])),
            tonic.transforms.ToFrame(sensor_size=(model_conf["in-size"],model_conf["in-size"],2),time_window=time_window),
            torch.from_numpy,
        ])


    cache_path=str(f"/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/exp/exp_gesture_1/test-cache/gesture-window{time_window}")
    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    testset=tonic.DiskCachedDataset(testset,cache_path=cache_path)
    test_loader = DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)
    #<< scalingしたデータの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> scalingしたデータを通す >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    out_s_scaled=[] # [testnum x N x T x xdim]
    out_v_scaled=[] # [testnum x N x T x xdim]
    with torch.no_grad():
        print_terminal(f"Forwarding x{timescale} scale input"+"-"*500)
        for inputs, targets in tqdm(test_loader):

            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
            inputs[inputs>0]=1.0

            if sequence>0 and inputs.shape[0]>sequence*timescale: #configでシーケンスが指定された場合はその長さに切り取る
                inputs=inputs[:int(sequence*timescale)]

            if not model_conf["type"]=="dynamic-snn":
                out_s, out_i, out_v = model(inputs) #[T x N x xdim]
            else:
                out_s, out_i, out_v = model.dynamic_forward(
                    s=inputs,scale_predictor=scale_predictor
                )
            out_s_scaled.append(out_s.permute(1,0,-1)) # [N x T x xdim]
            out_v_scaled.append(out_v.permute(1,0,-1)) # [N x T x xdim]

        # print(f"out_s_base length: {len(out_s_base)}, {out_s_base[0].shape}")
        out_s_scaled=torch.cat(out_s_scaled,dim=0) # [N x T x xdim]
        out_v_scaled=torch.cat(out_v_scaled,dim=0) # [N x T x xdim]
    print(f"out_s_scaled: {out_s_scaled.shape}")
    #<< scalingしたデータを通す <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print(f"ideal v:{out_v_ideal.shape}, scaled v:{out_v_scaled.shape}")
    

    def drop_mse(v_ideal,v_scaled,droprate):
        """
        :param v_ideal: [N x T x xdim]
        :param v_scaled: [N x T x xdim]
        :param droprate: 
        :return mse: scalr
        """
        mse=torch.mean((v_ideal-v_scaled)**2,dim=(1,2)) #バッチだけ残す
        # ランダムにシャッフル
        mse = mse[torch.randperm(mse.size(0))]

        return torch.mean(mse[:int(mse.shape[0]*(1-droprate))]).item()
    
    mse_v_list=[drop_mse(out_v_ideal,out_v_scaled,args.droprate) for _ in range(testnum)]
    mse_v_mean=np.mean(mse_v_list)
    mse_v_std=np.std(mse_v_list)
    print(f"mse_v_mean: {mse_v_mean}, mse_v_std: {mse_v_std}")


    mse_s_list=[drop_mse(out_s_ideal,out_s_scaled,args.droprate) for _ in range(testnum)]
    mse_s_mean=np.mean(mse_s_list)
    mse_s_std=np.std(mse_s_list)
    print(f"mse_s_mean: {mse_s_mean}, mse_s_std: {mse_s_std}")


    plot_batch_element(out_v_ideal,out_v_scaled,0,Path(args.saveto)/"v_0.png")
    plot_spike_element(out_s_ideal,out_s_scaled,0,Path(args.saveto)/"s_0.png")
    plot_firingrate_element(out_s_base,out_s_scaled,0,50,Path(args.saveto)/"fr_0.png")
    plot_firingrate_element(out_s_base,out_s_scaled,0,50,Path(args.saveto)/"fr_0.svg")
    save_dict2json(
        vars(args),saveto=Path(args.saveto)/"args.json"
    )

    result={
        "model":model_conf["type"],
        "datatype":train_conf["datatype"],
        "time-scale":args.timescale,
        "mse_v_mean":mse_v_mean,
        "mse_v_std":mse_v_std,
        "mse_s_mean":mse_s_mean,
        "mse_s_std":mse_s_std,
    }
    save_dict2json(
        result,saveto=Path(args.saveto)/"result.json"
    )


if __name__=="__main__":
    main()