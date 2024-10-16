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
from torch.utils.data import Dataset


from src.utils import load_yaml,print_terminal,calculate_accuracy
from src.model import DynamicSNN

from encoder import encode2spike


def ft(t):

    freq1=1
    f1=np.sin(2*np.pi*freq1*t)

    freq2=2
    f2=np.sin(2*np.pi*freq2*t)

    f=(f1+f2)/2

    return f1


class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, labels):
        """
        :param inputs: 入力データ (torch tensor) [N x T x m]
        :param labels: ラベルデータ (torch tensor) [N x T x m]
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]



def create_windows(data, window, overlap=0.5):
    """
    時系列データをウィンドウに分割する関数
    :param data: 時系列データ (torch tensor) [T x m]
    :param window: ウィンドウサイズ
    :param overlap: オーバーラップ率
    :return: ウィンドウに分割されたデータ (torch tensor) [N x window x m]
    """
    T, m = data.shape
    step = window - int(window*overlap)
    num_windows = int((T - overlap) // step)
    
    windows = []
    for i in range(num_windows):
        start = i * step
        end = start + window
        window_data=data[start:end, :]
        if window_data.shape[0]<window: #ウィンドウサイズより小さい場合はおしまい
            break
        windows.append(window_data)
    
    return torch.stack(windows)


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス",default=Path(__file__).parent)
    parser.add_argument("--device",default=0,help="GPUの番号")
    args = parser.parse_args()


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf=conf["train"],conf["model"]

    epoch=train_conf["epoch"]
    iter_max=train_conf["iter"]
    save_interval=train_conf["save_interval"]
    minibatch=train_conf["batch"]
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        model=DynamicSNN(model_conf)
        criterion=torch.nn.MSELoss()
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    model.to(device)

    #[N x h x T]で与えると各時間に対してlinearかけられる
    outnet=torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=model_conf["out-size"],out_channels=model_conf["out-size"],kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=model_conf["out-size"],out_channels=1,kernel_size=1),
        torch.nn.Tanh() #一応-1~1にしとく
    )
    outnet.to(device)

    optim=torch.optim.Adam(list(model.parameters()) + list(outnet.parameters()),lr=train_conf["lr"])
    if train_conf["schedule-step"]>0:   
        scheduler=StepLR(optimizer=optim,step_size=train_conf["schedule-step"],gamma=train_conf["schedule-rate"])
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    sequence_length=2000
    t=np.linspace(0,30,sequence_length)
    f=torch.tensor(ft(t)).unsqueeze(-1) #周波数1のsin波

    window=200
    overlap=0.8
    train_in=create_windows(f,window,overlap) #[N x T x m]

    t_next=np.roll(t,-1)
    f_next=torch.tensor(ft(t_next)).unsqueeze(-1)
    df=f_next-f
    train_label=create_windows(df,window,overlap) #[N x T x m], 正解ラベルは差分
    # train_label=create_windows(f_next,window,overlap) #[N x T x m], 正解ラベルは値

    trainset=TimeSeriesDataset(train_in,train_label)
    train_loader=DataLoader(trainset,batch_size=minibatch,shuffle=True, drop_last=True)
    print(f"original shape: {f.shape}, windowed shape: {train_in.shape}")
    print(f"original shape: {f_next.shape}, windowed shape: {train_label.shape}")
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> 学習ループ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    threshold=np.linspace(-1,1,model_conf["in-size"])
    result=[]
    for e in range(epoch):

        model.train()
        outnet.train()
        it=0
        train_loss_list=[]
        for batch_idx, (inputs, targets) in enumerate(train_loader): #[N x T x x]
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)
            # print(f"inputs: {inputs.shape}, targets: {targets.shape}")

            in_spikes=encode2spike(inputs,threshold).to(device)
            # print(f"in_spikes: {in_spikes.shape}")
            in_spikes=in_spikes.squeeze() #[N x T x (cxm)]
            in_spikes=in_spikes.permute(1,0,2) #[T x N x (cxm)]

            out_s, out_i, out_v = model.forward(in_spikes) #[T x N x h]
            # print(f"out_s: {out_s.shape}, out_i: {out_i.shape}, out_v: {out_v.shape}")

            out_v=out_v.permute(1,2,0) #[N x h x T]
            out=outnet(out_v) #[N x m x T]
            loss:torch.Tensor = criterion(out.permute(0,2,1), targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss_list.append(loss.item())

            print(f"Epoch [{e+1}/{epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            it+=1
            if iter_max>0 and it>iter_max:
                break

        print(f"targets : {targets[:,-1].to('cpu').detach().numpy().flatten()}")
        print(f"out: {out[:,:,-1].to('cpu').detach().numpy().flatten()}")
        print(f"out v sample: {out_v.to('cpu').detach().numpy()[0,:-1].flatten()}")
        print(f"out s sample: {out_s.to('cpu').detach().numpy()[-1,0].flatten()}")
            
        if train_conf["schedule-step"]>0:   
            scheduler.step() #学習率の更新

        # Save model checkpoint
        if (e + 1) % save_interval == 0:
            torch.save(model.state_dict(), resultpath / f"model_epoch_{e+1}.pth")
            torch.save(outnet.state_dict(), resultpath / f"outnet_epoch_{e+1}.pth")

        result.append([
            e,
            np.mean(train_loss_list), np.std(train_loss_list),
        ])
        result_db = pd.DataFrame(
            result, 
            columns=["epoch", "train_loss_mean", "train_loss_std"]
        )
        result_db.to_csv(resultpath / "training_results.csv", index=False)
        # Plot and save curves
    torch.save(model.state_dict(), resultpath / f"model_final.pth")
    torch.save(outnet.state_dict(), resultpath / f"outnet_final.pth")
    #<< 学習ループ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   


if __name__=="__main__":
    main()