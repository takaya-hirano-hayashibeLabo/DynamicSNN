"""
event_countでframeに変換してみる
"""

from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision
import tonic
import argparse
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


from src.model import DynamicCSNN,CSNN
from src.model import IFEncoder
# from src.model.dynamic_snn import DynamicSNN

def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # numpy.ndarrayをTensorに変換
    inputs = [torch.tensor(input) for input in inputs]
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor



import matplotlib.pyplot as plt


def plot_lif_states(lif_states, savepath:Path,batch_index=0, filename_prefix="lif_state"):
    """
    Plots the time series graphs for current, volt, and outspike for each layer in lif_states
    for a specified batch element.

    :param lif_states: Dictionary containing the states for each layer.
    :param batch_index: Index of the batch element to plot.
    :param filename_prefix: Prefix for the saved plot files.
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for layer_name, states in lif_states.items():
        outspike = states['outspike'].flatten(start_dim=2).cpu().numpy()
        current = states['current'].flatten(start_dim=2).cpu().numpy()
        volt = states['volt'].flatten(start_dim=2).cpu().numpy()

        timesteps, _, dim = current.shape

        plt.figure(figsize=(15, 15))

        for element in range(dim):
            # Plot current
            plt.subplot(3, 1, 1)
            plt.plot(range(timesteps), current[:, batch_index, element], label=f'Current Element {element}')
            plt.title(f'{layer_name} - Current Element {element} (Batch {batch_index})')
            plt.xlabel('Time Step')
            plt.ylabel('Current')
            plt.grid(True)
            plt.legend()

            # Plot volt
            plt.subplot(3,1,2)
            plt.plot(range(timesteps), volt[:, batch_index, element], label=f'Voltage Element {element}')
            plt.title(f'{layer_name} - Voltage Element {element} (Batch {batch_index})')
            plt.xlabel('Time Step')
            plt.ylabel('Voltage')
            plt.grid(True)
            plt.legend()

            # Plot outspike
            plt.subplot(3,1,3)
            plt.scatter(range(timesteps), outspike[:, batch_index, element]*(element+1), label=f'Outspike Element {element}')
            plt.title(f'{layer_name} - Outspike Element {element} (Batch {batch_index})')
            plt.xlabel('Time Step')
            plt.ylabel('Outspike')
            plt.ylim([0.5,dim+0.5])
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(savepath / f"{filename_prefix}_{layer_name}_batch{batch_index}.png")
        plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scale_type",default="real")
    parser.add_argument("--target",default="dyna-snn")
    args=parser.parse_args()

    relativepath="20241007.dyna-snn.eventcount/"
    resdir=Path(__file__).parent/f"{relativepath}"
    resdir.mkdir(exist_ok=True)


    device = "cuda:0"
    with open(Path(args.target)/'conf.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config["model"]["type"]=="dynamic-snn":
        model=DynamicCSNN(conf=config["model"])
    model.to(device)
    model.eval()
    print(model)

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval()

    a=3
    thr=50
    if_encoder=IFEncoder(threshold=thr,reset=0)

    datapath=ROOT/"original-data"
    event_count=100
    insize=4
    batch_size=5
    time_sequence=30
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),event_count=event_count),
        torch.from_numpy,
        ])

    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
    cachepath=ROOT/f"cache-data/gesture/eventcount{event_count}-insize{insize}"
    if os.path.exists(cachepath):
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)
    # base_in=np.array([testset[i][0][:time_sequence] for i in range(batch_size)])
    base_in,_=next(iter(testloader))
    base_in=base_in[:,:time_sequence]
    # plot_histogram(base_in,title="basein",filename=relativepath+"basein")
    base_in[base_in>0]=1.0 #スパイククリップ
    # base_in=if_encoder(base_in)
    base_in=torch.Tensor(base_in).permute(1,0,2,3,4)
    with torch.no_grad():
        base_s,_,base_v=model.forward(base_in.to(device).to(torch.float))
    print(f"input shape: {base_in.shape}, out spike shape: {base_s.shape}")

    lif_states=model.dynamic_forward_v1_with_lifstate(
        base_in.to(device).to(torch.float),
        a=torch.Tensor([1 for _ in range(base_in.shape[0])])
        )
    plot_lif_states(lif_states,resdir/f"lifstates_thr{thr}_a{a}/base")


    scale_type=args.scale_type

    #>> real #>>>>>>>>>>>>>>>>>
    alpha=1
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),event_count=int(event_count*a)),
        torch.from_numpy,
    ])
    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
    cachepath=ROOT/f"cache-data/gesture/eventcount{event_count}-insize{insize}"
    # if os.path.exists(cachepath): #cacheが悪さをしていました
    #     testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader_scaled=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)

    # scaled_in=np.array([testset[i][0][:int(a*time_sequence)] for i in range(batch_size)])
    scaled_in,_=next(iter(testloader_scaled))
    scaled_in=scaled_in[:,:int(a*time_sequence)]
    # plot_histogram(scaled_in,title="scaledin",filename=relativepath+"scaledin")
    scaled_in[scaled_in>0]=1.0 #スパイククリップ
    # scaled_in=if_encoder(scaled_in)
    scaled_in=torch.Tensor(scaled_in).permute(1,0,2,3,4)

    with torch.no_grad():
        lif_states=model.dynamic_forward_v1_with_lifstate(
            scaled_in.to(device).to(torch.float),
            a=torch.Tensor([alpha*a for _ in range(scaled_in.shape[0])])
            )
        plot_lif_states(lif_states,resdir/f"lifstates_thr{thr}_a{a}/real")



    #>> ideal >>>>>>>>>>>>>>>>>>>>>>>
    alpha=1.0
    input_size = config["model"]["in-size"]
    in_channel=config["model"]["in-channel"]
    scaled_in = torch.zeros(size=(int(a * time_sequence), batch_size, in_channel,input_size,input_size)).to(device)
    for t in range(time_sequence):
        scaled_index = int(a * t)
        if scaled_index < scaled_in.shape[0]:
            scaled_in[scaled_index] = base_in[t]

    with torch.no_grad():
        lif_states=model.dynamic_forward_v1_with_lifstate(
            scaled_in.to(device).to(torch.float),
            a=torch.Tensor([alpha*a for _ in range(scaled_in.shape[0])])
            )
        plot_lif_states(lif_states,resdir/f"lifstates_thr{thr}_a{a}/ideal")


if __name__=="__main__":
    main()