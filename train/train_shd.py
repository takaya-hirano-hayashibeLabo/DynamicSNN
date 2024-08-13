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


from src.utils import load_yaml,print_terminal,calculate_accuracy
from src.model import DynamicSNN,SNN,LSTM


def plot_and_save_curves(result, resultpath, epoch):
    df = pd.DataFrame(result, columns=["epoch", "train_loss_mean", "train_loss_std", "train_acc_mean", "train_acc_std", "val_loss_mean", "val_loss_std", "val_acc_mean", "val_acc_std"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axes[0].errorbar(df['epoch'], df['train_loss_mean'], yerr=df['train_loss_std'], label='Train Loss')
    axes[0].errorbar(df['epoch'], df['val_loss_mean'], yerr=df['val_loss_std'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curve')

    # Plot Accuracy
    axes[1].errorbar(df['epoch'], df['train_acc_mean'], yerr=df['train_acc_std'], label='Train Accuracy')
    axes[1].errorbar(df['epoch'], df['val_acc_mean'], yerr=df['val_acc_std'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig(resultpath / f'train_curves.png')
    plt.close()

def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス")
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
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="snn".casefold():
        model=SNN(model_conf)
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="lstm".casefold():
        model=LSTM(model_conf)
        criterion=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    model.to(device)
    optim=torch.optim.Adam(model.parameters(),lr=train_conf["lr"])
    if train_conf["schedule-step"]>0:   
        scheduler=StepLR(optimizer=optim,step_size=train_conf["schedule-step"],gamma=train_conf["schedule-rate"])
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=train_conf["time-window"]
    transform=torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size,time_window=time_window),
        torch.from_numpy,
    ])

    trainset=tonic.datasets.SHD(save_to=ROOT/"original-data",train=True,transform=transform)
    testset=tonic.datasets.SHD(save_to=ROOT/"original-data",train=False,transform=transform)


    train_loader = DataLoader(trainset, batch_size=minibatch, shuffle=True,collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn)
    print("done")
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> 学習ループ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    result=[]
    for e in range(epoch):

        model.train()
        it=0
        train_loss_list=[]
        train_acc_list=[]
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
            inputs=torch.squeeze(inputs) #いらないチャンネル次元を削除
            outputs = model.forward(inputs)
            loss:torch.Tensor = criterion(outputs, targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss_list.append(loss.item())

            if "snn".casefold() in model_conf["type"]:
                train_acc_list.append(SF.accuracy_rate(outputs,targets))
            else:
                train_acc_list.append(calculate_accuracy(outputs,targets))

            print(f"Epoch [{e+1}/{epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            it+=1
            if iter_max>0 and it>iter_max:
                break
            
        if train_conf["schedule-step"]>0:   
            scheduler.step() #学習率の更新

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = []
            test_acc_list=[]
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
                inputs=torch.squeeze(inputs) #いらないチャンネル次元を削除
                outputs = model(inputs)
                val_loss.append(criterion(outputs, targets).item())
                if "snn".casefold() in model_conf["type"]:
                    test_acc_list.append(SF.accuracy_rate(outputs,targets))
                else:
                    test_acc_list.append(calculate_accuracy(outputs,targets))

            print(f"Validation Loss after Epoch [{e+1}/{epoch}]: {np.mean(val_loss):.4f}")

        # Save model checkpoint
        if (e + 1) % save_interval == 0:
            torch.save(model.state_dict(), resultpath / f"model_epoch_{e+1}.pth")

        result.append([
            e,
            np.mean(train_loss_list), np.std(train_loss_list),
            np.mean(train_acc_list), np.std(train_acc_list),
            np.mean(val_loss),np.std(val_loss),
            np.mean(test_acc_list), np.std(test_acc_list)
        ])
        result_db = pd.DataFrame(
            result, 
            columns=["epoch", "train_loss_mean", "train_loss_std", "train_acc_mean", "train_acc_std", "val_loss_mean","val_loss_std", "val_acc_mean", "val_acc_std"]
        )
        result_db.to_csv(resultpath / "training_results.csv", index=False)
        # Plot and save curves
        plot_and_save_curves(result, resultpath, e + 1)
    torch.save(model.state_dict(), resultpath / f"model_final.pth")
    #<< 学習ループ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   


if __name__=="__main__":
    main()