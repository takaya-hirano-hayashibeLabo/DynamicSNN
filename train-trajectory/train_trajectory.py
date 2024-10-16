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


from src.utils import load_yaml,print_terminal,calculate_accuracy,Pool2DTransform,save_dict2json,create_windows
from src.model import DynamicSNN,ContinuousSNN
from src.model import ThresholdEncoder


def plot_and_save_curves(result, resultpath, epoch):
    # Adjust the columns to match the data structure
    df = pd.DataFrame(result, columns=["epoch", "datetime", "train_loss_mean", "train_loss_std", "val_loss_mean", "val_loss_std"])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot Loss
    ax.errorbar(df['epoch'], df['train_loss_mean'], yerr=df['train_loss_std'], label='Train Loss')
    ax.errorbar(df['epoch'], df['val_loss_mean'], yerr=df['val_loss_std'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Loss Curve')

    plt.tight_layout()
    plt.savefig(resultpath / f'train_curves.png')
    plt.close()


def plot_trajectory(outputs, targets, batch_idx, n, resultpath):
    """
    Plots the predicted and true trajectories for the n-th sample in the batch.
    
    :param outputs: The model's predictions for the batch.
    :param targets: The true labels for the batch.
    :param batch_idx: The index of the current batch.
    :param n: The index of the sample within the batch to plot.
    :param resultpath: The path to save the plot.
    """
    if n >= outputs.size(0):
        print(f"Sample index {n} is out of range for batch size {outputs.size(0)}.")
        return

    # Convert tensors to numpy for plotting
    predicted_trajectory = outputs[n].cpu().numpy()
    true_trajectory = targets[n].cpu().numpy()

    plt.figure(figsize=(10, 5))

    # Plot Predicted and True Trajectories
    plt.plot(predicted_trajectory, label='Predicted Trajectory', linestyle='-', color='blue', alpha=0.7, linewidth=2)
    plt.plot(true_trajectory, label='True Trajectory', linestyle='--', color='orange', alpha=0.7, linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Trajectory Value')
    plt.title(f'Batch {batch_idx}, Sample {n} Predicted vs True Trajectory')
    plt.legend()

    plt.tight_layout()
    plt.savefig(resultpath / f'trajectory_batch_{batch_idx}_sample_{n}.png')
    plt.close()


def plot_frequency_and_power_spectrum(data, column_name, resultpath):
    """
    Plots the frequency and power spectrum of the given data column.
    
    :param data: The data column to analyze.
    :param column_name: The name of the column.
    :param resultpath: The path to save the plot.
    """
    # Perform FFT
    fft_result = np.fft.fft(data)
    power_spectrum = np.abs(fft_result) ** 2
    frequencies = np.fft.fftfreq(len(data))

    # Plot Frequency and Power Spectrum
    plt.figure(figsize=(12, 6))

    # Plot Power Spectrum
    plt.subplot(2, 1, 1)
    plt.plot(frequencies, power_spectrum, label='Power Spectrum', color='blue')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title(f'{column_name} Power Spectrum')
    plt.legend()

    # Plot Frequency Spectrum (Magnitude)
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, np.abs(fft_result), label='Frequency Spectrum', color='orange')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title(f'{column_name} Frequency Spectrum')
    plt.legend()

    plt.tight_layout()
    plt.savefig(resultpath / f'{column_name}_frequency_power_spectrum.png')
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
    parser.add_argument('--target', type=str,help="configのあるパス", required=True)
    parser.add_argument("--device",default=0,help="GPUの番号")
    args = parser.parse_args()


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath/"models"):
        os.makedirs(resultpath/"models")

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf,encoder_conf=conf["train"],conf["model"],conf["encoder"]

    epoch=train_conf["epoch"]
    iter_max=train_conf["iter"]
    save_interval=train_conf["save_interval"]
    minibatch=train_conf["batch"]
    base_sequence=train_conf["sequence"]
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        time_model=DynamicSNN(model_conf)
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    model=ContinuousSNN(conf["output-model"],time_model)
    criterion=torch.nn.MSELoss() #連続値推論なのでMSE
    print(model)
    model.to(device)
    optim=torch.optim.Adam(model.parameters(),lr=train_conf["lr"])
    if train_conf["schedule-step"]>0:   
        scheduler=StepLR(optimizer=optim,step_size=train_conf["schedule-step"],gamma=train_conf["schedule-rate"])
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

    # # 各列を最大値と最小値で正規化
    # datasets_normalized = (datasets - datasets.min()) / (datasets.max() - datasets.min())
    # print(datasets_normalized)
    # datasets_normalized.to_csv(datapath.parent/"datasets_norm.csv",index=False)

    # # 各列の差分を計算し、最小値を出力
    # diffs = datasets_normalized.diff().iloc[1:]  # 最初の行は NaN になるので除外
    # min_diffs = diffs.min()
    # print("Minimum differences for each column:")
    # print(min_diffs)

    input_labels=["joint0","joint1","joint2","joint3","joint4","joint5"]
    input_datas=datasets[input_labels]
    input_max=input_datas.max()
    input_max.name="max"
    input_min=input_datas.min()
    input_min.name="min"
    input_nrm_params=pd.concat([input_max,input_min],axis=1)
    input_nrm_params.index = input_labels
    input_nrm_params.to_csv(resultpath/"input_nrm_params.csv", index=True)  # Ensure index is saved

    input_nrm_datas=2*((input_datas-input_min)/(input_max-input_min))[1:].values - 1 #入力データの正規化
    input_nrm_datas=create_windows(
        torch.Tensor(input_nrm_datas),
        window=base_sequence,
        overlap=0.95
    )

    target_datas=datasets[["target_x","target_y"]]

    # Apply to each column in target_datas
    for column in target_datas.columns:
        plot_frequency_and_power_spectrum(target_datas[column].diff().iloc[1:].dropna().values, column, resultpath)

    target_diff_datas=target_datas.diff().iloc[1:] #最初の行はNaNになるので除外
    target_diff_max=target_diff_datas.max()
    target_diff_max.name="max"
    target_diff_min=target_diff_datas.min()
    target_diff_min.name="min"
    target_diff_nrm_params=pd.concat([target_diff_max,target_diff_min],axis=1)
    target_diff_nrm_params.index=["target_x","target_y"]
    target_diff_nrm_params.to_csv(resultpath/"target_diff_nrm_params.csv",index=True)

    target_diff_datas=2*((target_diff_datas-target_diff_min)/(target_diff_max-target_diff_min))[1:].values - 1 #入力データの正規化

    target_diff_datas=create_windows(
        torch.Tensor(target_diff_datas),
        window=base_sequence,
        overlap=0.95
    )
    print("input nrm shape:",input_nrm_datas.shape,"target diff shape:",target_diff_datas.shape, "[N x T x m]")

    data_indices=np.arange(input_nrm_datas.shape[0])
    np.random.shuffle(data_indices)
    split_index=int(0.8*len(data_indices))
    train_indices=torch.tensor(data_indices[:split_index])
    test_indices=torch.tensor(data_indices[split_index:])

    train_inputs=input_nrm_datas[train_indices]
    train_targets=target_diff_datas[train_indices]
    test_inputs=input_nrm_datas[test_indices]
    test_targets=target_diff_datas[test_indices]

    print("train inputs shape:",train_inputs.shape,"train targets shape:",train_targets.shape)
    print("test inputs shape:",test_inputs.shape,"test targets shape:",test_targets.shape)

    train_dataset=torch.utils.data.TensorDataset(train_inputs,train_targets)
    test_dataset=torch.utils.data.TensorDataset(test_inputs,test_targets)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=minibatch,shuffle=True,drop_last=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=minibatch,shuffle=False)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> 学習ループ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    result=[]
    best_score={"mean":0.0, "std":0.0}
    for e in range(epoch):
        model.train()
        train_loss_list=[]
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs:torch.Tensor=encoder(inputs) # [N x T x resolution x xdim]
            inputs=inputs.flatten(start_dim=2) # [N x T x resolution*xdim]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            model.clip_gradients() #勾配クリッピング
            optim.step()
            optim.zero_grad()
            train_loss_list.append(loss.item())

        train_loss_mean=np.mean(train_loss_list)
        train_loss_std=np.std(train_loss_list)
        print(f"Epoch [{e+1}/{epoch}], Loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}")

        # Validation step
        model.eval()
        test_loss_list=[]
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs:torch.Tensor=encoder(inputs) # [N x T x resolution x xdim]
                inputs=inputs.flatten(start_dim=2) # [N x T x resolution*xdim]
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss_list.append(loss.item())

            # Plot the trajectory for the n-th sample in the batch
                n = 0  # Change this to the desired sample index
                plot_trajectory(outputs, targets, batch_idx, n, resultpath)

        test_loss_mean=np.mean(test_loss_list)
        test_loss_std=np.std(test_loss_list)
        print(f"Validation Loss after Epoch [{e+1}/{epoch}]: {test_loss_mean:.4f} ± {test_loss_std:.4f}")

        # Save model checkpoint
        if test_loss_mean<best_score["mean"]:
            best_score["mean"]=test_loss_mean
            best_score["std"]=test_loss_std
            best_score["epoch"]=e
            save_dict2json(best_score,resultpath/f"models/best-score.json")
            torch.save(model.state_dict(),resultpath/f"models/model_best.pth")

        result.append([
            e,
            datetime.now(),
            train_loss_mean, train_loss_std,
            test_loss_mean, test_loss_std
        ])

        result_db = pd.DataFrame(
            result, 
            columns=["epoch","datetime","train_loss_mean", "train_loss_std", "val_loss_mean","val_loss_std"]    
        )
        result_db.to_csv(resultpath / "training_results.csv", index=False)
        # Plot and save curves
        plot_and_save_curves(result, resultpath, e + 1)

        if (e + 1) % save_interval == 0:
            torch.save(model.state_dict(), resultpath / f"models/model_epoch_{e+1}.pth")

    torch.save(model.state_dict(), resultpath / f"models/model_final.pth")
    #<< 学習ループ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   


if __name__=="__main__":
    main()