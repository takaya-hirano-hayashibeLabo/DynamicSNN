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
import seaborn as sns
from math import floor

from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video
from src.model import DynamicCSNN,CSNN,DynamicResCSNN,ResNetLSTM,ResCSNN

def plot_histogram(data, title, xlabel, ylabel, save_path):
    plt.figure()
    sns.histplot(data, bins=50, kde=True, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス")
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--saveto",required=True,help="結果を保存するディレクトリ")
    parser.add_argument("--modelname",default="model_best.pth",help="モデルのファイル名")
    args = parser.parse_args()

    if not os.path.exists(Path(args.saveto) ):
        os.makedirs(Path(args.saveto))

    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf=conf["train"],conf["model"]
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=DynamicResCSNN(model_conf)
        else:
            model=DynamicCSNN(model_conf)
    elif model_conf["type"]=="snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=ResCSNN(model_conf)
        else:
            model=CSNN(model_conf)
    elif model_conf["type"]=="lstm".casefold():
        model=ResNetLSTM(model_conf)
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    tau_org=model.get_tau()
    tau_org_flatten=[]
    for key,tau in tau_org.items():
        tau_org_flatten+=list(torch.flatten(tau).to("cpu").numpy())
    
    modelname=args.modelname if ".pth" in args.modelname else args.modelname+".pth"
    model.load_state_dict(torch.load(Path(args.target)/f"result/{modelname}",map_location=device))
    model.eval() #別にしなくてもいい

    tau_trained=model.get_tau()
    tau_trained_flatten=[]
    for key,tau in tau_trained.items():
        tau_trained_flatten+=list(torch.flatten(tau).to("cpu").numpy())

    # ヒストグラムを描画
    plot_histogram(tau_org_flatten, 'Original Tau Histogram', 'Tau Value', 'Frequency', Path(args.saveto) / "tau_org_histogram.png")
    plot_histogram(tau_trained_flatten, 'Trained Tau Histogram', 'Tau Value', 'Frequency', Path(args.saveto) / "tau_trained_histogram.png")

if __name__=="__main__":
    main()