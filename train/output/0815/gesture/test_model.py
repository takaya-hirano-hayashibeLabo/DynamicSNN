import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent
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


from src.utils import load_yaml,print_terminal,calculate_accuracy,Pool2DTransform,save_dict2json
from src.model import DynamicCSNN,CSNN,DynamicResCSNN, ResCSNN, ResNetLSTM


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
    args = parser.parse_args()

    print(f"\nTarget model: {args.target}")


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
    sequence=train_conf["sequence"] #時系列のタイムシーケンス
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=train_conf["time-window"]
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


    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    testset=tonic.DiskCachedDataset(testset,cache_path=str(Path(args.target)/"cache/test"))
    test_loader = DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3)
    # print("done")
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    modelfiles = [file for file in os.listdir(Path(args.target) / "result") if "pth" in file]
    scores=[]
    best_score={"model":"","mean":0.0, "std":0.0}

    for i,modelfile in enumerate(modelfiles):
        print_terminal(f"[{i+1}/{len(modelfiles)}] test model@{modelfile}"+"-"*1000)


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
        # print(model)
        model.to(device)
        model.load_state_dict(torch.load(Path(args.target)/f"result/{modelfile}",map_location=device))



        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = []
            test_acc_list=[]
            for inputs, targets in tqdm(test_loader):
                inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
                inputs[inputs>0]=1.0

                if sequence>0 and inputs.shape[0]>sequence: #configでシーケンスが指定された場合はその長さに切り取る
                    inputs=inputs[:sequence]
                
                outputs = model(inputs)
                val_loss.append(criterion(outputs, targets).item())
                if "snn".casefold() in model_conf["type"]:
                    test_acc_list.append(SF.accuracy_rate(outputs,targets))
                else:
                    test_acc_list.append(calculate_accuracy(outputs,targets))


            acc_mean,acc_std=np.mean(test_acc_list),np.std(test_acc_list)
            print(f"Test ACC: {acc_mean:.4f}±{acc_std:.4f}")
            if acc_mean>best_score["mean"]: #テスト最高スコアのモデルを保存
                best_score["mean"]=acc_mean
                best_score["std"]=acc_std
                best_score["model"]=modelfile
                save_dict2json(best_score,resultpath/f"best-evalscore.json")

        scores.append([
            modelfile, acc_mean, acc_std
        ])

    scores=pd.DataFrame(scores,columns=["model","acc","std"])
    scores.to_csv(resultpath/"eval_score.csv",index=False)


if __name__=="__main__":
    main()