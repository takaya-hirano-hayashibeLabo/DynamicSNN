"""
速度倍率の変化が、スパイク(デルタ関数)のタイムスケール倍になるようなエンコーダを作る
"""

from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
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
import pandas as pd


from src.model import DynamicCSNN,CSNN
from src.utils import Pool2DTransform,print_terminal
from src.model import DiffEncoder, DirectCSNNEncoder
# from src.model.dynamic_snn import DynamicSNN



def plot_results(s_ideal, s_real ,filename):

    plt.figure(figsize=(12, 4))
    T=s_ideal.shape[0]

    plt.subplot(2,1,1)
    spike_times =torch.flatten(s_ideal,start_dim=2).cpu().numpy()[:,0]
    print(s_ideal.shape,spike_times.shape)
    plt.imshow(spike_times.T,interpolation="nearest", label=f'Ideal In Spikes',cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Ideal In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,T)


    plt.subplot(2,1,2)
    spike_times =torch.flatten(s_real,start_dim=2).cpu().numpy()[:,0]
    plt.imshow(spike_times.T,interpolation="nearest", cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Actual In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,T)


    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")


def main():

    device = "cuda:0"
    with open('conf.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config["model"]["type"]=="dynamic-snn":
        model=DynamicCSNN(conf=config["model"])
    model.to(device)
    model.eval()
    print(model)

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval()

    scores=[]

    datapath=ROOT/"original-data"
    sensor_size=(2,128,128)
    time_window=10000
    resize=(32,32)
    batch_size=5
    time_sequence=100
    poolsize=int(sensor_size[-1]/resize[-1])
    pooltype="avg"
    thre=0.5

    # encoder=DiffEncoder(
    #     threshold=thre
    # )
    
    encoder=DirectCSNNEncoder(config["model"])
    encoder.to(device)


    for time_window in range(1000,30000,1000):
            
            print_terminal(contents=f"\ntime window : {time_window}"+"-"*500)
            transform=torchvision.transforms.Compose([
                # Event2Frame(sensor_size,time_window),
                tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
                torch.from_numpy,
                # torchvision.transforms.Resize(resize,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                Pool2DTransform(pool_size=poolsize,pool_type=pooltype)
                ])

            testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
            
            base_in=np.array([testset[i][0][:time_sequence] for i in range(batch_size)])
            base_in_max=np.max(base_in)
            base_in/=base_in_max
            # base_in[base_in>0]=1

            #>> encoding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            with torch.no_grad():
                base_in=torch.stack(
                    [encoder.step_forward(torch.Tensor(base_in[:,t]).to(device)) for t in range(base_in.shape[1])],dim=0
                )
                encoder.reset_state()
            #>> encoding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # base_in=torch.Tensor(base_in).permute(1,0,2,3,4)

            with torch.no_grad():
                base_s,base_v=model(base_in.to(device))
            print(f"input shape: {base_in.shape}, out spike shape: {base_s.shape}")


            a=5

            alpha=1
            transform=torchvision.transforms.Compose([
                # Event2Frame(sensor_size,time_window),
                tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=int(time_window/a)),
                torch.from_numpy,
                # torchvision.transforms.Resize(resize,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                Pool2DTransform(pool_size=poolsize,pool_type=pooltype)
                ])
            testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
            in_real=np.array([testset[i][0][:int(a*time_sequence)] for i in range(batch_size)])
            in_max=np.max(in_real)
            print(f"scaled in max: {in_max}")
            in_real/=base_in_max
            # in_real[in_real>0]=1

            #>> encoding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>            
            with torch.no_grad():
                in_real=torch.stack(
                    [encoder.step_forward(torch.Tensor(in_real[:,t]).to(device)) for t in range(in_real.shape[1])],dim=0
                )
                encoder.reset_state()
            #>> encoding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            in_real=in_real.to(device)
            # in_real=torch.Tensor(in_real).permute(1,0,2,3,4)

            alpha=1.0
            input_size = config["model"]["in-size"]
            in_channel=config["model"]["in-channel"]
            in_ideal = torch.zeros(size=(int(a * time_sequence), batch_size, in_channel,input_size,input_size)).to(device)
            for t in range(time_sequence):
                in_idealdex = int(a * t)
                if in_idealdex < in_ideal.shape[0]:
                    in_ideal[in_idealdex] = base_in[t]


            bias=12
            plot_size=4
            plot_results(
                s_ideal=in_ideal[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
                s_real=in_real[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
                filename=f"img/dvsgesture_{time_window}"
            )

            loss_in_spike=torch.mean((in_real.to(device)-in_ideal.to(device))**2).item()


            with torch.no_grad():
                s_real_dyna,v_real_dyna=model.dynamic_forward_v1(
                    in_real,
                    a=[a for _ in range(in_real.shape[0])]
                )

                s_real_sta,v_real_sta=model.forward(in_real)

            v_resampled=F.interpolate(base_v.permute(1,2,0), size=in_ideal.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
            s_resampled=F.interpolate(base_s.permute(1,2,0), size=in_ideal.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
            loss_outsp_org_dyna=torch.mean((s_resampled-s_real_dyna)**2).item()
            loss_outsp_org_sta=torch.mean((s_resampled-s_real_sta)**2).item()
            loss_outv_org_dyna=torch.mean((v_resampled-v_real_dyna)**2).item()
            loss_outv_org_sta=torch.mean((v_resampled-v_real_sta)**2).item()


            scores.append(
                [time_window,loss_in_spike,loss_outsp_org_dyna,loss_outsp_org_sta,
                 loss_outv_org_dyna,loss_outv_org_sta]
            )


    columns=["time-window","loss-inSpike","loss-outSpike-Org vs Dyna",
             "loss-outSpike-Org vs Sta","loss-outVolt-Org vs Dyna","loss-outVolt-Org vs Sta"]
    scores=pd.DataFrame(scores,columns=columns)
    scores.to_csv(Path(__file__).parent/"score.csv",index=False)


if __name__=="__main__":
    main()