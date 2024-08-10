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


from src.model import DynamicCSNN,CSNN
from src.utils import Pool2DTransform
# from src.model.dynamic_snn import DynamicSNN



def plot_results(s1,s2,v1,v3 , filename):

    plt.figure(figsize=(12, 12))
    T=s1.shape[0]
    T_scaled=v3.shape[0]

    plt.subplot(5,1,1)
    spike_times =torch.flatten(s1,start_dim=2).cpu().numpy()[:,0]
    print(s1.shape,spike_times.shape)
    plt.imshow(spike_times.T,interpolation="nearest", label=f'Base Spike Dim',cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(5,1,2)
    v1_ =torch.flatten(v1,start_dim=2).cpu().numpy()[:,0]  
    plt.imshow(v1_.T,interpolation="nearest",cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(5,1,3)
    v1_ =torch.flatten(v1,start_dim=2)
    v1_resampled=F.interpolate(v1_.permute(1,2,0), size=s2.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    v1_resampled=v1_resampled.cpu().numpy()[:,0]
    im3 = plt.imshow(v1_resampled.T,interpolation="nearest",cmap="viridis",aspect="auto")
    cbar3 = plt.colorbar(im3)  # Save the colorbar to use its limits
    plt.title('Scaled Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(5,1,4)
    spike_times =torch.flatten(s2,start_dim=2).cpu().numpy()[:,0]
    plt.imshow(spike_times.T,interpolation="nearest", cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Scaled In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(5,1,5)
    v3_ =torch.flatten(v3,start_dim=2).cpu().numpy()[:,0]
    im6 = plt.imshow(v3_.T,interpolation="nearest", cmap="viridis",aspect="auto", vmin=im3.get_clim()[0], vmax=im3.get_clim()[1])
    plt.colorbar(im6)  # Use the same colorbar limits
    plt.title('Original Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))
    

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")


    loss_org= np.mean((v1_resampled-v3_)**2)

    print(f"loss org : {loss_org.item()}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scale_type",default="real")
    parser.add_argument("--target",default="snn")
    args=parser.parse_args()

    device = "cuda:0"
    with open(Path(args.target)/'conf.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config["model"]["type"]=="snn":
        model=CSNN(conf=config["model"])
    model.to(device)
    model.eval()
    print(model)

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval()

    datapath=ROOT/"original-data"
    sensor_size=(2,128,128)
    time_window=10000
    resize=(32,32)
    batch_size=5
    time_sequence=100
    poolsize=int(sensor_size[-1]/resize[-1])
    transform=torchvision.transforms.Compose([
        # Event2Frame(sensor_size,time_window),
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
        torch.from_numpy,
        # torchvision.transforms.Resize(resize,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        Pool2DTransform(pool_size=poolsize)
        ])

    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
    
    base_in=np.array([testset[i][0][:time_sequence] for i in range(batch_size)])
    base_in[base_in>0]=1
    base_in=torch.Tensor(base_in).permute(1,0,2,3,4)
    with torch.no_grad():
        base_s,base_v=model(base_in.to(device))
    print(f"input shape: {base_in.shape}, out spike shape: {base_s.shape}")



    a=5

    scale_type=args.scale_type

    if scale_type=="real":
        transform=torchvision.transforms.Compose([
            # Event2Frame(sensor_size,time_window),
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=int(time_window/a)),
            torch.from_numpy,
            # torchvision.transforms.Resize(resize,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            Pool2DTransform(pool_size=poolsize)
            ])
        testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
        scaled_in=np.array([testset[i][0][:int(a*time_sequence)] for i in range(batch_size)])
        scaled_in[scaled_in>0]=1
        scaled_in=torch.Tensor(scaled_in).permute(1,0,2,3,4)

    elif scale_type=="ideal":
        input_size = config["model"]["in-size"]
        in_channel=config["model"]["in-channel"]
        scaled_in = torch.zeros(size=(int(a * time_sequence), batch_size, in_channel,input_size,input_size)).to(device)
        for t in range(time_sequence):
            scaled_index = int(a * t)
            if scaled_index < scaled_in.shape[0]:
                scaled_in[scaled_index] = base_in[t]

    with torch.no_grad():
        org_s,org_v=model.forward(scaled_in.to(device))
        print(f"scaled input shape: {scaled_in.shape}, original out spike shape: {org_s.shape}")


    bias=12
    plot_size=4
    plot_results(
        s1=base_in[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
        s2=scaled_in[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
        v1=base_v,
        v3=org_v,
        filename=f"{args.target}/dvsgesture_{scale_type}"
    )

    base_fr=torch.mean(base_s,dim=0)
    org_fr=torch.mean(org_s,dim=0)

    print(f"base fr: \n{base_fr}\n")
    print(f"org fr: \n{org_fr}\n")

    loss_sp_org=torch.mean((base_fr-org_fr)**2).item()
    print(f"loss spike org: {loss_sp_org}")

if __name__=="__main__":
    main()