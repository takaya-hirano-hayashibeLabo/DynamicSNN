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
import pandas as pd


from src.utils import load_yaml,load_hdf5,print_terminal,CustomDataset,calculate_accuracy
from src.dynamic_snn import DynamicSNN
from src.snn import SNN 
from src.lstm import LSTM

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
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="snn".casefold():
        model=SNN(model_conf)
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="lstm".casefold():
        model=LSTM(model_conf)
        criterion=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    
    model.to(device)
    optim=torch.optim.Adam(model.parameters(),lr=train_conf["lr"])
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print_terminal(contents="preparing dataset...")
    datapath=train_conf["datapath"]
    train_files=[f"{datapath}/train/{file}" for file in os.listdir(f"{datapath}/train")]
    in_train,target_train=load_hdf5(train_files,num_workers=32) #[batch x time-sequence x ...], [batch]
    test_files=[f"{datapath}/test/{file}" for file in os.listdir(f"{datapath}/test")]
    in_test,target_test=load_hdf5(test_files,num_workers=32) #[batch x time-sequence x ...], [batch]
    
    # ここでデータをfloat型に変換
    in_train = torch.Tensor(np.array(in_train)).to(torch.float)
    in_test =  torch.Tensor(np.array(in_test)).to(torch.float)

    # ターゲットデータをLong型に変換
    target_train = torch.Tensor(np.array(target_train)).to(torch.long)
    target_test =  torch.Tensor(np.array(target_test )).to(torch.long)

    train_dataset = CustomDataset(in_train, target_train)
    test_dataset = CustomDataset(in_test, target_test)

    train_loader = DataLoader(train_dataset, batch_size=minibatch, shuffle=True)
    test_loader = DataLoader(test_dataset,   batch_size=minibatch, shuffle=False)
    print("done")
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> 学習ループ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    result=[]
    for e in range(epoch):

        model.train()
        it=0
        train_loss_list=[]
        train_acc_list=[]
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])), targets.to(device)
            outputs = model.forward(inputs)
            loss:torch.Tensor = criterion(outputs, targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss_list.append(loss.item())

            if "snn".casefold() in model_conf["type"]:
                train_acc_list.append(SF.accuracy_rate(outputs,targets))
            else:
                train_acc_list.append(calculate_accuracy(outputs,targets))

            print(f"Epoch [{e+1}/{epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # it+=1
            # if iter_max>0 and it>iter_max:
            #     break

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            test_acc_list=[]
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                if "snn".casefold() in model_conf["type"]:
                    test_acc_list.append(SF.accuracy_rate(outputs,targets))
                else:
                    test_acc_list.append(calculate_accuracy(outputs,targets))

            val_loss /= len(test_loader)
            print(f"Validation Loss after Epoch [{e+1}/{epoch}]: {val_loss:.4f}")

        # Save model checkpoint
        if (e + 1) % save_interval == 0:
            torch.save(model.state_dict(), resultpath / f"model_epoch_{e+1}.pth")

        result.append([e,np.mean(train_loss_list),np.mean(train_acc_list),val_loss,np.mean(test_acc_list)])
        result_db = pd.DataFrame(
            result, 
            columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )
        result_db.to_csv(resultpath / "training_results.csv", index=False)
    torch.save(model.state_dict(), resultpath / f"model_final.pth")
    #<< 学習ループ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   


if __name__=="__main__":
    main()