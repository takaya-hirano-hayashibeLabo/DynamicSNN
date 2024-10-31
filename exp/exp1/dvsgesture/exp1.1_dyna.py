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

def plot_results(s1,s2,v1, v2,v3 , filename):

    plt.figure(figsize=(12, 12))
    T=s1.shape[0]
    T_scaled=v2.shape[0]

    plt.subplot(6,1,1)
    spike_times =torch.flatten(s1,start_dim=2).cpu().numpy()[:,0]
    print(s1.shape,spike_times.shape)
    plt.imshow(spike_times.T,interpolation="nearest", label=f'Base Spike Dim',cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,2)
    v1_ =torch.flatten(v1,start_dim=2).cpu().numpy()[:,0]  
    plt.imshow(v1_.T,interpolation="nearest",cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,3)
    v1_ =torch.flatten(v1,start_dim=2)
    v1_resampled=F.interpolate(v1_.permute(1,2,0), size=s2.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    v1_resampled=v1_resampled.cpu().numpy()[:,0]
    im3 = plt.imshow(v1_resampled.T,interpolation="nearest",cmap="viridis",aspect="auto")
    cbar3 = plt.colorbar(im3)  # Save the colorbar to use its limits
    plt.title('Scaled Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,4)
    spike_times =torch.flatten(s2,start_dim=2).cpu().numpy()[:,0]
    plt.imshow(spike_times.T,interpolation="nearest", cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Scaled In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,5)
    v2_ =torch.flatten(v2,start_dim=2).cpu().numpy()[:,0]
    im5 = plt.imshow(v2_.T,interpolation="nearest", cmap="viridis",aspect="auto", vmin=im3.get_clim()[0], vmax=im3.get_clim()[1])
    plt.colorbar(im5)  # Use the same colorbar limits
    plt.title('Dynamic CSNN Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,6)
    v3_ =torch.flatten(v3,start_dim=2).cpu().numpy()[:,0]
    im6 = plt.imshow(v3_.T,interpolation="nearest", cmap="viridis",aspect="auto", vmin=im3.get_clim()[0], vmax=im3.get_clim()[1])
    plt.colorbar(im6)  # Use the same colorbar limits
    plt.title('Original Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))
    

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")


    loss_dyna=np.mean((v1_resampled-v2_)**2)
    loss_org= np.mean((v1_resampled-v3_)**2)

    print(f"loss dyna : {loss_dyna.item()}, loss org : {loss_org.item()}, loss org/loss dyna: {(loss_org/loss_dyna).item()}")


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
    args=parser.parse_args()

    relativepath="20241029.dyna-snn_exp1.1/"
    resdir=Path(__file__).parent/f"{relativepath}"
    resdir.mkdir(exist_ok=True)


    device = "cuda:0"
    with open(Path(args.target)/'conf.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config["model"]["type"]=="dynamic-snn":
        # model=DynamicCSNN(conf=config["model"])
        model=DynamicResCSNN(conf=config["model"])
    modelname="model_best.pth"
    model.load_state_dict(torch.load(Path(args.target)/f"result/{modelname}",map_location=device))
    model.to(device)
    model.eval()
    model.output_mem=True #membrane potentialを出力するようにする
    print(model)

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval()

    threshold=25
    if_encoder=IFEncoder(threshold=threshold)

    datapath=ROOT/"original-data"
    time_window=3000
    insize=32
    batch_size=3
    time_sequence=300
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
        torch.from_numpy,
        ])

    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)
    cachepath=ROOT/f"cache-data/gesture/window{time_window}-insize{insize}"
    # if os.path.exists(cachepath):
    #     testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)
    # base_in=np.array([testset[i][0][:time_sequence] for i in range(batch_size)])
    base_in,_=next(iter(testloader))
    base_in=base_in[:,:time_sequence]
    # plot_histogram(base_in,title="basein",filename=relativepath+"basein")
    print("save plot histogram")
    # exit(1)


    # base_in[base_in>0]=1.0 #スパイククリップ
    base_in=if_encoder(base_in)
    base_in=torch.Tensor(base_in).permute(1,0,2,3,4)
    with torch.no_grad():
        base_s,_,base_v=model.forward(base_in.to(device).to(torch.float))
    print(f"input shape: {base_in.shape}, input sum :{torch.sum(base_in)}, out spike shape: {base_s.shape}, out spike sum: {torch.sum(base_s)}")



    a=3
    scale_type=args.scale_type

    if scale_type=="real":
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
        plot_histogram(scaled_in,title="scaledin",filename=relativepath+"scaledin")
        # scaled_in[scaled_in>0]=1.0 #スパイククリップ
        scaled_in=if_encoder(scaled_in)
        scaled_in=torch.Tensor(scaled_in).permute(1,0,2,3,4)

    elif scale_type=="ideal":
        alpha=1.0
        input_size = config["model"]["in-size"]
        in_channel=config["model"]["in-channel"]
        scaled_in = torch.zeros(size=(int(a * time_sequence), batch_size, in_channel,input_size,input_size)).to(device)
        for t in range(time_sequence):
            scaled_index = int(a * t)
            if scaled_index < scaled_in.shape[0]:
                scaled_in[scaled_index] = base_in[t]

    with torch.no_grad():
        org_s,_,org_v=model.forward(scaled_in.to(device).to(torch.float))
        print(f"scaled input shape: {scaled_in.shape}, scaled input sum :{torch.sum(scaled_in)}, original out spike shape: {org_s.shape}, original out spike sum: {torch.sum(org_s)}")

        scaled_s,_,scaled_v=model.dynamic_forward_v1(
            scaled_in.to(device).to(torch.float),
            a=torch.Tensor([alpha*a for _ in range(scaled_in.shape[0])])
            )
        print(f"scaled input shape: {scaled_in.shape}, scaled input sum :{torch.sum(scaled_in)}, Dynamic out spike shape: {scaled_s.shape}, Dynamic out spike sum: {torch.sum(scaled_s)}")

    bias=0
    plot_size=-1


    plot_results(
        s1=base_in[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
        s2=scaled_in[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
        v1=base_v,
        v2=scaled_v,
        v3=org_v,
        filename=f"20241029.dyna-snn_exp1.1/dvsgesture_{scale_type}"
    )

    base_fr=torch.mean(base_s,dim=0)
    org_fr=torch.mean(org_s,dim=0)
    scaled_fr=torch.mean(scaled_s,dim=0)

    # print(f"base fr: \n{base_fr}\n")
    # print(f"org fr: \n{org_fr}\n")
    # print(f"dynamic fr: \n{scaled_fr}\n")

    loss_sp_org=torch.mean((base_fr-org_fr)**2).item()
    loss_sp_dyna=torch.mean((base_fr-scaled_fr)**2).item()
    print(f"loss spike org: {loss_sp_org}, loss spike dyna: {loss_sp_dyna}")

if __name__=="__main__":
    main()