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
    parser.add_argument("--saveto",required=True,help="結果を保存するディレクトリ")
    parser.add_argument("--modelname",default="model_best.pth",help="モデルのファイル名")
    parser.add_argument("--is_video", action='store_true')
    parser.add_argument("--timescale1",type=int,default=1,help="前半の速度")
    parser.add_argument("--timescale2",type=int,default=1,help="後半の速度")
    parser.add_argument("--testnum",type=int,default=5,help="stdを求めるために何回testするか")
    parser.add_argument("--test_droprate",type=float,default=0.2,help="testデータにランダム性を持たせるために, 1minibatchごとにdropするrate")
    args = parser.parse_args()


    testnum=args.testnum
    test_droprate=args.test_droprate
    timescale1,timescale2=args.timescale1,args.timescale2
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
    time_window=int(train_conf["time-window"]/timescale1) if "time-window" in train_conf.keys() else int(train_conf["timewindow"]/timescale1)
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
    testset1=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    testset1=tonic.DiskCachedDataset(testset1,cache_path=cache_path)
    test_loader1 = DataLoader(testset1,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)



    time_window=int(train_conf["time-window"]/timescale2) if "time-window" in train_conf.keys() else int(train_conf["timewindow"]/timescale2)
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
    testset2=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
    testset2=tonic.DiskCachedDataset(testset2,cache_path=cache_path)
    test_loader2 = DataLoader(testset2,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=True)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # Validation step
    print_terminal(f"eval model: {model_conf['type']}@ timescale: ({timescale1}, {timescale2})"+"-"*500)
    model.eval()
    test_list=[]

    with torch.no_grad():
        for i_test in range(testnum):
            test_acc_list_i=[]
            for (inputs1, targets1), (inputs2, targets2) in zip(tqdm(test_loader1), tqdm(test_loader2)):
                inputs1, targets1 = inputs1.to(device).permute((1, 0, *[i + 2 for i in range(inputs1.ndim - 2)])).to(torch.float), targets1.to(device)
                inputs2, targets2 = inputs2.to(device).permute((1, 0, *[i + 2 for i in range(inputs2.ndim - 2)])).to(torch.float), targets2.to(device)
                

                seq1,seq2=int(sequence/2),int(sequence/2)
                inputs=torch.cat(
                    [inputs1[:seq1*timescale1],inputs2[seq1*timescale2:seq1*timescale2+seq2*timescale2]], dim=0
                )

                #>> testデータにランダム性を持たせるために確率的にdropする >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                mask=(torch.rand(inputs.shape[1])>test_droprate).to(device)
                inputs,targets1=inputs[:,mask],targets1[mask]
                #<< testデータにランダム性を持たせるために確率的にdropする <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                inputs[inputs>0]=1.0


                if not model_conf["type"]=="dynamic-snn":
                    outputs = model(inputs)
                else:

                    outputs=model.dynamic_forward(
                        inputs,scale_predictor
                    )

                if "snn".casefold() in model_conf["type"]:
                    test_acc_list_i.append(SF.accuracy_rate(outputs,targets1))
                else:
                    test_acc_list_i.append(calculate_accuracy(outputs,targets1))
            acc_mean,acc_std=np.mean(test_acc_list_i),np.std(test_acc_list_i)
            print(f"[{i_test+1}/{testnum}] ACC {acc_mean:.2f} ± {acc_std:.2f}")

            test_list.append(acc_mean)

    print_terminal(f"{testnum} Trial Result: ACC {np.mean(test_list):.4f}±{np.std(test_list):.4f}")
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
                file_name=f"train_input_label{targets1[i_frame]}",
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
        "timescale1":args.timescale1,
        "timescale2":args.timescale2,
        "acc_mean":np.mean(test_list),
        "acc_std":np.std(test_list)
    }
    save_dict2json(
        result,saveto=Path(args.saveto)/"result.json"
    )


if __name__=="__main__":
    main()