"""
IFencoderを使ってみる
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


from src.model import DynamicCSNN,CSNN,DynamicResCSNN
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


def get_avtive_elm_idx(in_spikes:torch.Tensor,batch_idx=0,view_idx_num=10):
    """
    :param in_spikes: [timestep x batch x xdim...]
    """
    in_spikes_batch_i=in_spikes[:,batch_idx]
    in_spikes_batch_i=torch.flatten(in_spikes_batch_i,start_dim=1)
    spike_activation=torch.mean(in_spikes_batch_i,dim=0)
    active_elm_idx=torch.argsort(spike_activation,dim=0,descending=True)[:view_idx_num]
    return active_elm_idx


def plot_in_spike(in_spikes:torch.Tensor,savepath:Path,filename_prefix="in_spike",batch_idx=0,start_elm_idx=0,end_elm_idx=-1,xlim=None):
    """
    :param in_spikes: [timestep x batch x xdim...]
    """
    in_spikes_batch_i=in_spikes[:,batch_idx]
    in_spikes_batch_i=torch.flatten(in_spikes_batch_i,start_dim=1)[:,start_elm_idx:end_elm_idx].to("cpu").detach().numpy()
    timesteps, dim = in_spikes_batch_i.shape

    plt.figure(figsize=(15, 15))
    for d in range(dim):
        plt.plot(range(timesteps),in_spikes_batch_i[:,d]*d,label=f"dim{d}",marker="o",linestyle="None")
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim([0.5,dim+0.5])
    plt.grid(True)
    plt.savefig(savepath/f"{filename_prefix}_batch{batch_idx}_start{start_elm_idx}_end{end_elm_idx}.png")
    plt.close()



def plot_lif_states(lif_states, savepath:Path, batch_index=0, filename_prefix="lif_state"):
    """
    Plots the time series graphs for current, volt, and outspike for the last layer in lif_states
    for a specified batch element.

    :param lif_states: Dictionary containing the states for each layer.
    :param batch_index: Index of the batch element to plot.
    :param filename_prefix: Prefix for the saved plot files.
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Get the last item in the lif_states dictionary
    last_layer_name, last_states = list(lif_states.items())[-1]
    
    outspike = last_states['outspike'].flatten(start_dim=2).cpu().numpy()
    current = last_states['current'].flatten(start_dim=2).cpu().numpy()
    volt = last_states['volt'].flatten(start_dim=2).cpu().numpy()

    timesteps, _, dim = current.shape

    plt.figure(figsize=(15, 15))

    for element in range(dim):
        # Plot current
        plt.subplot(3, 1, 1)
        plt.plot(range(timesteps), current[:, batch_index, element], label=f'Current Element {element}')
        plt.title(f'{last_layer_name} - Current Element {element} (Batch {batch_index})')
        plt.xlabel('Time Step')
        plt.ylabel('Current')
        plt.grid(True)
        plt.legend()

        # Plot volt
        plt.subplot(3, 1, 2)
        plt.plot(range(timesteps), volt[:, batch_index, element], label=f'Voltage Element {element}')
        plt.title(f'{last_layer_name} - Voltage Element {element} (Batch {batch_index})')
        plt.xlabel('Time Step')
        plt.ylabel('Voltage')
        plt.grid(True)
        plt.legend()

        # Plot outspike
        plt.subplot(3, 1, 3)
        plt.scatter(range(timesteps), outspike[:, batch_index, element] * (element + 1), label=f'Outspike Element {element}')
        plt.title(f'{last_layer_name} - Outspike Element {element} (Batch {batch_index})')
        plt.xlabel('Time Step')
        plt.ylabel('Outspike')
        plt.ylim([0.5, dim + 0.5])
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(savepath / f"{filename_prefix}_{last_layer_name}_batch{batch_index}.png")
    plt.close()


def plot_histogram(data, title, filename):
    plt.figure()
    plt.hist(data.flatten(), bins=10, alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(Path(__file__).parent / f"{filename}.png")
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scale_type",default="real")
    parser.add_argument("--target",default="dyna-snn")
    parser.add_argument("--device",default=0)
    args=parser.parse_args()

    relativepath="20241120.dynasnn/"
    resdir=Path(__file__).parent/f"{relativepath}"
    resdir.mkdir(exist_ok=True)


    device = f"cuda:{args.device}"
    # with open(Path(args.target)/'conf_in4.yml', 'r') as file:
    with open(Path(args.target)/'conf.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config["model"]["type"]=="dynamic-snn":
        model=DynamicResCSNN(conf=config["model"])
    model.to(device)
    model.eval()
    print(model)

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval(
    a=5
    thr=50
    if_encoder=IFEncoder(threshold=thr)

    datapath=ROOT/"original-data"
    time_window=3000
    insize=config["model"]["in-size"]
    batch_size=5
    time_sequence=50
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
        torch.from_numpy,
    ])

    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
    cachepath=ROOT/f"cache-data/gesture/window{time_window}-insize{insize}"
    if os.path.exists(cachepath):
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)
    # base_in=np.array([testset[i][0][:time_sequence] for i in range(batch_size)])
    base_in,_=next(iter(testloader))
    base_in=base_in[:,:time_sequence]
    plot_histogram(base_in,title="basein",filename=relativepath+"basein")
    # base_in[base_in>0]=1.0 #スパイククリップ
    base_in=if_encoder(base_in)
    base_in=torch.Tensor(base_in).permute(1,0,2,3,4)
    active_elm_idx=get_avtive_elm_idx(base_in.to(device).to(torch.float))
    print(f"active_elm_idx: {active_elm_idx}")

    with torch.no_grad():
        base_s,_,base_v=model.forward(base_in.to(device).to(torch.float))
    # print(f"input shape: {base_in.shape}, out spike shape: {base_s.shape}")
    print(f"base_in spike counts: {base_in.sum()}")

    lif_states=model.dynamic_forward_v1_with_lifstate(
        base_in.to(device).to(torch.float),
        a=torch.Tensor([1 for _ in range(base_in.shape[0])])
        )
    plot_lif_states(lif_states,resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}/base")

    plot_elm_size=128
    plot_in_spike(base_in,resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}/base",
                  start_elm_idx=active_elm_idx[0],end_elm_idx=active_elm_idx[0]+plot_elm_size,xlim=(0,time_sequence))
    scale_type=args.scale_type

    #>> real #>>>>>>>>>>>>>>>>>
    alpha=1
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=int(time_window/a)),
        torch.from_numpy,
    ])
    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
    cachepath=ROOT/f"cache-data/gesture/window{time_window}-insize{insize}"
    # if os.path.exists(cachepath): #cacheが悪さをしていました
    #     testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader_scaled=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)

    # scaled_in=np.array([testset[i][0][:int(a*time_sequence)] for i in range(batch_size)])
    scaled_in,_=next(iter(testloader_scaled))
    scaled_in=scaled_in[:,:int(a*time_sequence)]
    # plot_histogram(scaled_in,title="scaledin",filename=relativepath+"scaledin")
    # scaled_in[scaled_in>0]=1.0 #スパイククリップ
    scaled_in=if_encoder(scaled_in)
    scaled_in=torch.Tensor(scaled_in).permute(1,0,2,3,4)
    print(f"scaled_in spike counts: {scaled_in.sum()}")

    with torch.no_grad():
        lif_states=model.dynamic_forward_v1_with_lifstate(
            scaled_in.to(device).to(torch.float),
            a=torch.Tensor([1 for _ in range(scaled_in.shape[0])])
            )
        plot_lif_states(lif_states,resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}/real")
    plot_in_spike(scaled_in,resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}/real",
                  start_elm_idx=active_elm_idx[0],end_elm_idx=active_elm_idx[0]+plot_elm_size,xlim=(0,int(a*time_sequence)))



    #>> ideal >>>>>>>>>>>>>>>>>>>>>>>
    alpha=1.0
    input_size = config["model"]["in-size"]
    in_channel=config["model"]["in-channel"]
    scaled_in_ideal = torch.zeros(size=(int(a * time_sequence), batch_size, in_channel,input_size,input_size)).to(device)
    for t in range(time_sequence):
        scaled_index = int(a * t)
        if scaled_index < scaled_in.shape[0]:
            scaled_in_ideal[scaled_index] = base_in[t]

    with torch.no_grad():
        lif_states=model.dynamic_forward_v1_with_lifstate(
            scaled_in_ideal.to(device).to(torch.float),
            a=torch.Tensor([alpha*a for _ in range(scaled_in_ideal.shape[0])])
            )
        plot_lif_states(lif_states,resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}/ideal")
    plot_in_spike(scaled_in_ideal,resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}/ideal",
                  start_elm_idx=active_elm_idx[0],end_elm_idx=active_elm_idx[0]+plot_elm_size,xlim=(0,int(a*time_sequence)))

    actual_error=torch.sum(torch.abs(scaled_in.to("cpu").detach()-scaled_in_ideal.to("cpu").detach()))
    print(f"actual_error: {actual_error}")

if __name__=="__main__":
    main()