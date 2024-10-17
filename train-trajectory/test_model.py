import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent
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
from math import floor
from datetime import datetime
from tqdm import tqdm


from src.utils import load_yaml,print_terminal,calculate_accuracy,Pool2DTransform,save_dict2json,create_windows
from src.model import DynamicSNN,ContinuousSNN
from src.model import ThresholdEncoder




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help="configのあるパス", default=Path(__file__).parent)
    parser.add_argument("--device",default=0,help="GPUの番号")
    args = parser.parse_args()

    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath/"models"):
        os.makedirs(resultpath/"models")

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf,encoder_conf=conf["train"],conf["model"],conf["encoder"]
    base_sequence=train_conf["sequence"]
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        time_model=DynamicSNN(model_conf)
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    model=ContinuousSNN(conf["output-model"],time_model)
    print(model)
    model.to(device)
    model.load_state_dict(torch.load(resultpath/"models/model_best.pth"))
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> encoderの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if encoder_conf["type"].casefold()=="thr":
        encoder=ThresholdEncoder(
            thr_max=encoder_conf["thr-max"],thr_min=encoder_conf["thr-min"],
            resolution=encoder_conf["resolution"],device=device
        )
    else:
        raise ValueError(f"encoder type {encoder_conf['type']} is not supportated...")
    #<< encoderの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    datapath:Path=ROOT/train_conf["datapath"]
    datasets=pd.read_csv(datapath)
    # print(datasets)

    # input_labels=["joint0","joint1","joint2","joint3","joint4","joint5"]
    input_labels=[f"endpos_{label}" for label in ["x","y"]]
    input_datas=datasets[input_labels]
    input_max=input_datas.max()
    input_max.name="max"
    input_min=input_datas.min()
    input_min.name="min"

    print(f"in max,min: {input_max.values}, {input_min.values}")

    target_datas=datasets[["target_x","target_y"]]
    if conf["output-model"]["out-type"].casefold()=="velocity":
        target_datas=target_datas.diff().iloc[1:] #最初の行はNaNになるので除外
    elif conf["output-model"]["out-type"].casefold()=="position":
        target_datas=target_datas.iloc[1:]
    target_max=target_datas.max()
    target_max.name="max"
    target_min=target_datas.min()
    target_min.name="min"
    print(f"target max,min: {target_max.values}, {target_min.values}")
      
    model.eval()
    T=500
    n_head=100
    in_trajectory=[]
    for t in tqdm(range(T)):

        if t<n_head:
            in_trajectory.append(input_datas.values[t])
        else:
            in_x=np.array(in_trajectory)[-base_sequence:] if len(in_trajectory)>base_sequence else np.array(in_trajectory)
            in_x=2*(in_x-input_min.values)/(input_max.values-input_min.values)-1

            in_spike=encoder(torch.Tensor(in_x).unsqueeze(0).to(device))

            with torch.no_grad():
                out_nrm=model.forward(in_spike.flatten(start_dim=2))[0,-1].to("cpu").detach().numpy()
            out=0.5*(out_nrm+1)*(target_max.values-target_min.values)+target_min.values
            print(f"out nrm: {out_nrm}, out: {out}")

            next_state=in_trajectory[-1]+out #差分を足し合わせる
            in_trajectory.append(next_state)

    print(f"in trj shape: {np.array(in_trajectory).shape}")
    in_trj_db=pd.DataFrame(in_trajectory,columns=["target_x","target_y"])
    in_trj_db.to_csv(resultpath/"test_result.csv")


if __name__ == "__main__":
    main()