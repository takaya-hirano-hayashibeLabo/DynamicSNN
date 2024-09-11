import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
EXP=Path(__file__).parent
import sys
sys.path.append(str(ROOT))

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import torchvision
import tonic
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


from src.utils import print_terminal,save_dict2json


def plot_firing_rate(result_db,saveto):
    plt.figure(figsize=(10, 6))
    plt.plot(result_db['time-scale'], result_db['firing-rate'], marker='o')
    plt.xlabel('Time Scale')
    plt.ylabel('Firing Rate')
    plt.title('Firing Rate vs Time Scale')
    plt.grid(True)
    filename=saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else f"{str(saveto)}/result.png"
    plt.savefig(filename)
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
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--load_csv",action="store_true")
    args = parser.parse_args()


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(__file__).parent/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    minibatch=16
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    w=3000
    insize=32
    sequence=300

    config={
        "base-timewindow":w,
        "in-size":insize,
        "input-sequence":sequence,
        "minibatch":minibatch
    }
    save_dict2json(config,resultpath/"config.json")


    if not args.load_csv:
        window_range=[0.5,1,2]
        # window_range=range(1,20,2)
        result=[]
        for i,timescale in enumerate(window_range):
        
            time_window=int(w/timescale)
            if insize==128:
                transform=torchvision.transforms.Compose([
                    tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
                    tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
                    torch.from_numpy,
                ])
            else:
                transform=torchvision.transforms.Compose([
                    tonic.transforms.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
                    tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
                    torch.from_numpy,
                ])


            cache_path=str(Path(__file__).parent/f"test-cache/gesture-window{time_window}")
            testset=tonic.datasets.DVSGesture(save_to=ROOT/"original-data",train=False,transform=transform)
            testset=tonic.DiskCachedDataset(testset,cache_path=cache_path)
            test_loader = DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3)
            #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            # Validation step
            print_terminal(f"[{i+1}/{len(window_range)}]window: {time_window}@ time-scale: {timescale}"+"-"*500)
            with torch.no_grad():
                fr_list=[]
                for inputs, targets in tqdm(test_loader):
                    inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device) #[time-sequence x batch x xdim]
                    inputs[inputs>0]=1.0

                    if sequence>0 and inputs.shape[0]>sequence*timescale: #configでシーケンスが指定された場合はその長さに切り取る
                        inputs=inputs[:int(sequence*timescale)]

                    fr=torch.mean(inputs,dim=0) #時間方向に平均
                    fr=torch.mean(fr,dim=tuple([i+1 for i in range(fr.ndim-1)])) #空間方向に平均
                    fr=torch.mean(fr) #batch方向に平均

                    fr_list.append(fr.item())

                result.append([
                    time_window,timescale,np.mean(fr_list),np.std(fr_list)
                ])
            print_terminal(f"done\n")

            result_db=pd.DataFrame(result,columns=["time-window","time-scale","firing-rate","firing-rate_std"])
            result_db.to_csv(resultpath/"result.csv",index=False)

    elif args.load_csv:
        result_db=pd.read_csv(resultpath/"result.csv")

    plot_firing_rate(result_db,resultpath/"result.png")



    #>> 対数をとって線形回帰 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    X=np.log(result_db["firing-rate"].values.reshape(-1,1)+1e-10)
    Y=np.log(result_db["time-scale"].values+1e-10)
    model=LinearRegression()
    model.fit(X,Y)

    slope=model.coef_[0]
    intercept=model.intercept_

    test_x=[
        0.025047595777055797 ,0.012982142075677128 ,0.00885123570504434  ,
        0.006723269,0.005430870949674179 ,0.00454992544837296  ,
        0.003916250719853184 ,0.003449421,0.003068269,0.002760755
    ]
    plt.scatter(result_db["firing-rate"].values,result_db["time-scale"].values,label="test plot")
    plt.plot(np.array(test_x),np.exp(model.predict(np.log(test_x).reshape(-1,1))),label="regression",color="red")
    plt.xlabel('Firing Rate')
    plt.ylabel('Scale')
    plt.legend()
    plt.savefig(resultpath/"regression.png")
    plt.close()

    regression_result={
        "slope":slope,
        "intercept":intercept
    }
    save_dict2json(regression_result,resultpath/"regression-params.json")
    #<< 対数をとって線形回帰 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__=="__main__":
    main()