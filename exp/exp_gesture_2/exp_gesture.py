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


from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video
from src.model import DynamicCSNN,CSNN,DynamicResCSNN,ResNetLSTM,ResCSNN, ScalePredictor


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
    parser.add_argument("--delay",default=50,type=int,help="何倍に時間をスケールするか. time_scale=2でtime-windowが1/2になる.")
    parser.add_argument("--saveto",required=True,help="結果を保存するディレクトリ")
    parser.add_argument("--modelname",default="model_best.pth",help="モデルのファイル名")
    parser.add_argument("--is_video", action='store_true')
    # parser.add_argument("--timescale",default=)
    args = parser.parse_args()

    delay_step=args.delay
    if not os.path.exists(Path(args.saveto) ):
        os.makedirs(Path(args.saveto))


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf=conf["train"],conf["model"]

    # minibatch=train_conf["batch"]
    minibatch=16
    sequence=train_conf["sequence"] #時系列のタイムシーケンス
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=DynamicResCSNN(model_conf)
        else:
            model=DynamicCSNN(model_conf)
        scale_predictor=ScalePredictor(datatype="gesture")
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
    model_name=args.modelname if ".pth" in args.modelname else args.modelname+".pth"
    model.load_state_dict(torch.load(Path(args.target)/f"result/{model_name}",map_location=device))
    model.to(device)
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=int(train_conf["time-window"])
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


    cache_path=str(EXP/f"test-cache/gesture-window{time_window}")
    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    testset=tonic.DiskCachedDataset(testset,cache_path=cache_path)
    test_loader = DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # Validation step
    print_terminal(f"eval model: {model_conf['type']}@ delay: {delay_step}"+"-"*500)
    model.eval()
    fisrt_idx=150
    with torch.no_grad():
        val_loss = []
        test_acc_list=[]
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
            inputs[inputs>0]=1.0


            if sequence>0 and inputs.shape[0]>sequence: #configでシーケンスが指定された場合はその長さに切り取る
                inputs=inputs[:sequence]


            #>> delayを入れる >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if delay_step>0:
                delay=torch.where(
                    torch.rand(size=(delay_step,*inputs.size()[1:]))<0.005, #0.5%発火のノイズをdelaystep分噛ませる
                    1.0,0.0
                ).to(device)
                inputs=torch.cat([
                    inputs[:fisrt_idx],delay,inputs[fisrt_idx:]
                ],dim=0)
            #<< delayを入れる <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            if not model_conf["type"]=="dynamic-snn":

                outputs = model(inputs)
            else:

                # outputs=model(inputs)
                # a_first=[1 for _ in range(fisrt_idx)]
                # a_delay=[int(delay_step/5) for _ in range(delay_step)]
                # a_second=[1 for _ in range(inputs.shape[0]-fisrt_idx)]
                # a=a_first+a_delay+a_second
                # outputs=model.dynamic_forward_v1(
                #     inputs,a=a
                # )
                outputs=model.dynamic_forward(
                    inputs,scale_predictor
                )

            val_loss.append(criterion(outputs, targets).item())
            if "snn".casefold() in model_conf["type"]:
                test_acc_list.append(SF.accuracy_rate(outputs,targets))
            else:
                test_acc_list.append(calculate_accuracy(outputs,targets))
    acc_mean,acc_std=np.mean(test_acc_list),np.std(test_acc_list)
    print(f"ACC {acc_mean:.2f} ± {acc_std:.2f}")
    print_terminal(f"done\n")


    ##>> 入力確認 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.is_video:
        print_terminal(f"saveing sample videos...")
        video_size=320
        sample_num=5
        for i_frame in tqdm(range(sample_num)):
            frame_np=inputs[:,i_frame].to("cpu").detach().numpy()
            frame=1.5*frame_np[:,0]+0.5*frame_np[:,1]-1
            save_heatmap_video(
                frame,
                output_path=Path(args.saveto)/f"video",
                file_name=f"train_input_label{targets[i_frame]}",
                fps=60,scale=int(video_size/model_conf["in-size"])
            )
        print_terminal("done")
    ##<< 入力確認 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    save_dict2json(
        vars(args),saveto=Path(args.saveto)/"args.json"
    )

    result={
        "model":model_conf["type"],
        "datatype":train_conf["datatype"],
        "delay":args.delay,
        "acc_mean":acc_mean,
        "acc_std":acc_std
    }
    save_dict2json(
        result,saveto=Path(args.saveto)/"result.json"
    )


if __name__=="__main__":
    main()