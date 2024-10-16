"""
テスト項目
・速度変化がスパイクのタイムスケール変化に落とし込めるか？
=原理的な部分の調査
"""

import yaml 
from pathlib import Path
import sys
MODELDIR=Path(__file__).parent.parent.parent.parent
sys.path.append(str(MODELDIR))
ROOT=MODELDIR.parent
import torch
import matplotlib.pyplot as plt
import torchvision
import tonic
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import pandas as pd

from model.encoder import ThresholdEncoder
from utils import save_heatmap_video


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # numpy.ndarrayをTensorに変換
    inputs = [torch.tensor(input) for input in inputs]
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor



def save_spike_frames_histogram(spike_frames, filename="spike_frames_histogram.png"):
    plt.hist(spike_frames.flatten().cpu().numpy(), bins=50)
    plt.title('Spike Frames Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()

def maxpool(x):
    maxpooling=torch.nn.MaxPool2d(4)

    # MaxPoolを適用
    N, T, C, H, W = x.shape
    x = x.view(N * T, C, H, W)  # [batch*timestep, channel, h, w]
    x = maxpooling(x.to(torch.float))  # [batch*timestep, channel, h', w']
    x = x.view(N, T, C, x.shape[2], x.shape[3])  # [batch, timestep, channel, h', w']
    return x


def event2frame(x:torch.Tensor):
    """
    :param x :[N x T x 2 x w x h]
    """
    return 1.5*x[:,:,0] + 0.5*x[:,:,1]-1

def encout2frame(enc_out):
    """
    :param enc_out: [T x 3 x h x w]
    """
    frame=enc_out[:,0]*4 + enc_out[:,1]*2 + enc_out[:,2]
    return (frame/7)*2-1

def plot_sample(xout_sample, filename):
    for class_idx in range(xout_sample.shape[1]):
        x=np.arange(0,xout_sample.shape[0])
        y=xout_sample[:,class_idx].to("cpu").detach().numpy()*(class_idx+1)
        plt.scatter(x,y, label=f'Class {class_idx}', s=1)

    plt.xlabel('Time Step')
    plt.ylim([0.5,3.5])
    plt.ylabel('Class')
    plt.title('xout_sample Plot')
    plt.legend()
    plt.savefig(str(Path(__file__).parent/filename)+".png")
    plt.close()


def culculate_firingrate(spikes:torch.Tensor,timescale):
    """
    :param spikes: [T x c]
    """
    channel_size=spikes.shape[1]
    timewindow=int(30*timescale) #basewindowは30とする
    firingrate=[[None for _ in range(channel_size)] for _ in range(timewindow)]
    for t in range(timewindow, spikes.shape[0]):
        firingrate.append(
            list(torch.mean(spikes[t-timewindow:t].float(),dim=0).to("cpu").detach().numpy())
        )
    return np.array(firingrate)
    

def plot_fr(firingrate,filename):
    classnum=len(firingrate[0])
    T=len(firingrate)
    for class_idx in range(classnum):
        x=np.linspace(0,1,T)
        y=firingrate[:,class_idx]
        plt.plot(x,y, label=f'Class {class_idx}')

    plt.xlabel('Time Step')
    plt.ylim([0,1])
    plt.ylabel('firingrate')
    plt.title('firingrate')
    plt.legend()
    plt.savefig(str(Path(__file__).parent/filename)+".png")
    plt.close()

def plot_frame(frames,filename):
    classnum=len(frames[0])
    T=len(frames)
    for class_idx in range(classnum):
        x=np.arange(0,T)
        y=frames[:,class_idx]
        plt.vlines(x,0,y, label=f'Class {class_idx}')

    plt.xlabel('Time Step')
    plt.ylabel('frame val')
    plt.title('frame')
    plt.legend()
    plt.savefig(str(Path(__file__).parent/filename)+".png")
    plt.close()

def main():

    # Create a random input tensor
    insize=32
    batchsize=1


    #>> 普通のtimewindow >>
    a1=1
    time_window=int(3000/a1)
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
        torch.from_numpy,
    ])

    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)

    cachepath=ROOT/f"cache-data/gesture/window{time_window}-insize{insize}"
    if os.path.exists(cachepath):
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader1=DataLoader(testset,batch_size=batchsize,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)
    #>> 普通のtimewindow >>


    #>> スケール変えたtimewindow >>
    a2=5
    time_window=int(3000/a2)
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
        torch.from_numpy,
    ])

    testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)

    cachepath=ROOT/f"cache-data/gesture/window{time_window}-insize{insize}"
    if os.path.exists(cachepath):
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))
    testloader2=DataLoader(testset,batch_size=batchsize,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)
    #>> 普通のtimewindow >>



    #>> 閾値エンコーダ >>
    thr_enc=ThresholdEncoder(thr_max=0.75, thr_min=-0.75,resolution=3) #入力は [N x T x xdim]
    #>> 閾値エンコーダ >>


    base_sequence=50
    sequence=int(base_sequence*a1)
    xin1,_=next(iter(testloader1)) #[N x T x xdim]
    xin1[xin1>0]=1.0 #1にクリップ
    # xin1=maxpool(xin1)
    xout1=thr_enc(event2frame(xin1[:,:sequence]))[0]
    print("xout1 shape : ",xout1.shape)
    print(f"xin1 spike num: {torch.sum(xin1[:sequence])}, xout1 spike num: {torch.sum(xout1)}")

    sequence=int(base_sequence*a2)
    xin2,_=next(iter(testloader2)) #[N x T x xdim]
    xin2[xin2>0]=1.0 #1にクリップ
    # xin2=maxpool(xin2)
    xout2=thr_enc(event2frame(xin2[:,:sequence]))[0]
    print("xout2 shape : ",xout2.shape)
    print(f"xin2 spike num: {torch.sum(xin2[:sequence])}, xout2 spike num: {torch.sum(xout2)}")


    # >> 一番動きのあるピクセルを調査 >>
    sum_idx_pair=[]
    xtest=xout1.flatten(start_dim=2)
    for idx in range(xtest.shape[-1]):
        xtest_sample=xtest[:,:,idx]
        sum_idx_pair.append(
            [idx,torch.sum(xtest_sample)]
        )
    N=10
    top_N = sorted(sum_idx_pair, key=lambda x: x[1], reverse=True)[:N]
    top_N_db=pd.DataFrame(top_N,columns=["idx", "spike_num"])
    print(top_N_db)
    

    # save_heatmap_video(encout2frame(xout1).to("cpu").detach().numpy(),output_path=Path(__file__).parent,file_name=f"xout_a{a1}",frame_label_view=False)
    # save_heatmap_video(encout2frame(xout2).to("cpu").detach().numpy(),output_path=Path(__file__).parent,file_name=f"xout_a{a2}",frame_label_view=False)

    idx=746
    xout1_sample=xout1.flatten(start_dim=2)[:,:,idx]
    xout2_sample=xout2.flatten(start_dim=2)[:,:,idx]
    print(xout1_sample.shape)
    plot_sample(xout1_sample,filename=f"xout_a{a1}")
    plot_sample(xout2_sample,filename=f"xout_a{a2}")

    print(f"spike1 sum : {torch.sum(xout1_sample,dim=0)}")
    print(f"spike2 sum : {torch.sum(xout2_sample,dim=0)}")

    xout1_fr=culculate_firingrate(xout1_sample,a1)
    xout2_fr=culculate_firingrate(xout2_sample,a2)
    plot_fr(xout1_fr,f"xout_fr_a{a1}")
    plot_fr(xout2_fr,f"xout_fr_a{a2}")


    xin1_sample=xin1[0].flatten(start_dim=2)[:,:,idx][:int(base_sequence*a1)]
    xin2_sample=xin2[0].flatten(start_dim=2)[:,:,idx][:int(base_sequence*a2)]
    # print(xin1_sample)
    # print(xin1_sample.shape)
    xin1_fr=culculate_firingrate(xin1_sample,a1)
    xin2_fr=culculate_firingrate(xin2_sample,a2)
    plot_fr(xin1_fr,f"xin_fr_a{a1}")
    plot_fr(xin2_fr,f"xin_fr_a{a2}")
    plot_sample(xin1_sample,f"xin_a{a1}")
    plot_sample(xin2_sample,f"xin_a{a2}")
    print(f"xin1 sum : {torch.sum(xin1_sample,dim=0)}")
    print(f"xin2 sum : {torch.sum(xin2_sample,dim=0)}")

    print(event2frame(xin1_sample.unsqueeze(0)).numpy().shape)
    plot_frame(event2frame(xin1_sample.unsqueeze(0)).unsqueeze(-1).numpy()[0],f"xin_frame_a{a1}")
    plot_frame(event2frame(xin2_sample.unsqueeze(0)).unsqueeze(-1).numpy()[0],f"xin_frame_a{a2}")

if __name__=="__main__":
    main()