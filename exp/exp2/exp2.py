from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import os
from snntorch import functional as SF
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm


from src.dynamic_snn import DynamicSNN
from src.snn import SNN
from src.scale_predictor import ScalePredictor
from src.utils import load_yaml,load_hdf5,print_terminal,CustomDataset,scale_sequence,resample_scale

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス")
    parser.add_argument("--device",default=0,help="GPUの番号")
    args = parser.parse_args()
    

    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf=conf["train"],conf["model"]

    epoch=train_conf["epoch"]
    iter_max=train_conf["iter"]
    save_interval=train_conf["save_interval"]
    minibatch=train_conf["batch"]
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        model=DynamicSNN(model_conf)
        scale_predictor=ScalePredictor(datatype=train_conf["datatype"])
    elif model_conf["type"]=="snn".casefold():
        model=SNN(model_conf)
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    model.load_state_dict(torch.load((Path(args.target)/"result/model_final.pth")))
    model.to(device)
    model.eval()

    # # Debug model params
    # model_path = Path(args.target) / "result/model_final.pth"
    # state_dict = torch.load(model_path)
    # # Print the loaded parameters for debugging
    # for param_tensor in state_dict:
    #     print(f"{param_tensor}: {state_dict[param_tensor].size()}")
    #     print(state_dict[param_tensor])

    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_terminal(contents="preparing dataset...")
    datapath=train_conf["datapath"]
    test_files=[f"{datapath}/test/{file}" for file in os.listdir(f"{datapath}/test")]
    in_test,target_test=load_hdf5(test_files,num_workers=32) #[batch x time-sequence x ...], [batch]
    
    # ここでデータタイムスケールを変換
    a=[30 for _ in range(len(in_test[0])+1)] #スケールリスト
    # a=np.ones(len(in_test[0])+1) #スケールリスト
    # a[int(len(a)*1/3):int(len(a)*2/3)]=50
    in_test = scale_sequence(np.array(in_test),a=a,dt=model_conf["dt"])
    in_test =  torch.Tensor(np.array(in_test)).to(torch.float)
    a_resampled=resample_scale(a,target_length=in_test.shape[1])

    # ターゲットデータをLong型に変換
    target_test =  torch.Tensor(np.array(target_test )).to(torch.long)
    test_dataset = CustomDataset(in_test, target_test)
    test_loader = DataLoader(test_dataset,   batch_size=minibatch, shuffle=False)
    print("done")
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> テスト >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    with torch.no_grad():
        test_acc_list=[]
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])), targets.to(device)

            if model_conf["type"]=="dynamic-snn".casefold():
                outputs = model.dynamic_forward(inputs,scale_predictor)
                # outputs = model.forward(inputs)
            else:
                outputs = model(inputs)
            test_acc_list.append(SF.accuracy_rate(outputs,targets))

    print(f"Accuracy: {np.mean(test_acc_list)}")
    #<< テスト <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def test1():
    dt=0.1
    spike_times=[0.1,0.8, 1.6,2]
    a=1.5

    T=2
    Nt=int(T/dt)
    elapsed=0
    new_spike_times=[]
    for t in range(Nt+1):

        if t*dt<= spike_times[0] <(t+1)*dt:
            t_sp=spike_times.pop(0)
            new_t_sp=elapsed
            new_spike_times.append(new_t_sp)

        elapsed+=a*dt

        if len(spike_times)<1:
            break
    
    print(new_spike_times)


def test2():

    dt=0.1
    spike=np.array(
        [
            [1,1,0],
            [0,0,1],
            [1,1,0],
            [0,0,1],
            [1,1,0],
            [0,0,1],
        ]
    )
    # spike=np.array(
    #     [
    #         [[1,1,0],
    #         [0,0,1]],
    #         [[1,1,0],
    #         [0,0,1]],
    #         [[1,1,0],
    #         [0,0,1]],
    #         [[1,1,0],
    #         [0,0,1]],
    #         [[1,1,0],
    #         [0,0,1]],
    #     ]
    # )

    # timestamp, idx_x=spike2timestamp(spike,dt)
    # print(timestamp)
    # print(idx_x)

    a=3*np.ones(spike.shape[0]+1)
    scaled_sp=scale_sequence([spike],a,dt)[0]
    print(scaled_sp)

if __name__=="__main__":
    main()

