"""
IFencoderを使ってみる
"""

from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent
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
import re


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

def plot_and_save_lif_states(lif_states, save_dir='plots', zoom_tmax=10):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for key, value in lif_states.items():
        volt = value["volt"]
        current = value["current"]  # Assuming 'current' is also a key in the dictionary
        
        # Move tensors to CPU and convert to numpy
        volt = volt.cpu().numpy()
        current = current.cpu().numpy()
        
        # Squeeze the batch dimension
        volt = np.squeeze(volt, axis=1)
        current = np.squeeze(current, axis=1)
        
        # Flatten the xdim dimensions
        volt = volt.reshape(volt.shape[0], -1)
        current = current.reshape(current.shape[0], -1)
        
        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        
        # Plot volt
        for i in range(volt.shape[1]):
            axs[0].plot(volt[:, i], label=f'xdim {i}')
        axs[0].set_title(f'Voltage over Time for {key}')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('Voltage')
        # axs[0].legend(loc='upper right')
        
        # Plot current
        for i in range(current.shape[1]):
            axs[1].plot(current[:, i], label=f'xdim {i}')
        axs[1].set_title(f'Current over Time for {key}')
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel('Current')
        # axs[1].legend(loc='upper right')


        # Plot volt
        for i in range(volt.shape[1]):
            axs[2].plot(volt[:, i], label=f'xdim {i}')
        axs[2].set_title(f'Voltage over Time for {key}')
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylabel('Voltage')
        axs[2].set_xlim(0,zoom_tmax)
        # axs[0].legend(loc='upper right')
        
        # Plot current
        for i in range(current.shape[1]):
            axs[3].plot(current[:, i], label=f'xdim {i}')
        axs[3].set_title(f'Current over Time for {key}')
        axs[3].set_xlabel('Timestep')
        axs[3].set_ylabel('Current')
        axs[3].set_xlim(0,zoom_tmax)
        # axs[1].legend(loc='upper right')
        
        # Save the plot as a PNG file
        file_path = os.path.join(save_dir, f'{key}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",default="dyna-snn")
    parser.add_argument("--device",default=0,type=int)
    args=parser.parse_args()

    model_dir=re.sub("(.*)output/","",str(args.target))

    relativepath=f"20241030.dynasnn/{model_dir}"
    resdir=Path(__file__).parent/f"{relativepath}"
    if not os.path.exists(resdir):
        os.makedirs(resdir)


    device = f"cuda:{args.device}"
    # with open(Path(args.target)/'conf_in4.yml', 'r') as file:
    with open(Path(args.target)/'conf.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    model=DynamicResCSNN(conf=config["model"])
    model.load_state_dict(torch.load(Path(args.target)/f"result/model_best.pth",map_location=device))
    model.to(device)
    model.eval()
    print(model)

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval(
    a=1
    thr=100
    if_encoder=IFEncoder(threshold=thr)

    datapath=ROOT/"original-data"
    time_window=3000
    insize=config["model"]["in-size"]
    batch_size=1
    time_sequence=300

    #>> real #>>>>>>>>>>>>>>>>>
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

    for key,value in lif_states.items():
        print(key,value["volt"].shape)

    plot_and_save_lif_states(
        lif_states,save_dir=resdir/f"lifstates_insize{insize}_window{time_window}_thr{thr}_a{a}",
        zoom_tmax=10*a
        )

if __name__=="__main__":
    main()