import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import os
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm


def save_to_hdf5(file_path, data, target):
    """
    hdf5で圧縮して保存
    そうしないとTオーダのデータサイズになってしまう
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('events', data=data.numpy().astype(np.int8), compression='gzip')
        f.create_dataset('target', data=target)



def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--filename",default="xor")
    parser.add_argument("--sequence-size",type=int,default=500,help="全体の時系列長さ")
    parser.add_argument("--insize",default=12,type=int,help="入力次元数")
    parser.add_argument("--trainsize",default=1000,type=int,help="trainデータのサイズ")
    parser.add_argument("--testsize",default=200,type=int,help="testデータのサイズ")
    parser.add_argument("--empty-frame-rate",default=0.2,type=float,help="空のフレームの割合")
    args=parser.parse_args()


    savedir=(ROOT/f"train-data/{args.filename}")
    if not os.path.exists(savedir):
        os.makedirs(savedir)


    rate_high,rate_low,rate_none=0.8,0.3,0.05 #high->1, low->0, none->空
    rate_pattern=[
        {"in":[rate_high,rate_none,rate_high], "out":0},
        {"in":[rate_low,rate_none,rate_low],   "out":0},
        {"in":[rate_high,rate_none,rate_low],  "out":1},
        {"in":[rate_low,rate_none,rate_high],  "out":1},
    ]


    events,targets=[],[]

    datasize=args.trainsize+args.testsize
    in_size=args.insize

    T=args.sequence_size
    T_none=int(T*args.empty_frame_rate) #空のスパイクフレームを作る
    T1=int((T-T_none)/2) #前半の入力フレーム長さ
    T2=T-(T1+T_none) #後半の入力フレームの長さ
    T_list=[T1,T_none,T2]

    for pattern in rate_pattern:

        r1,rn,r2=pattern["in"]

        input_spikes=[]
        for iT in range(3): #前半入力フレーム・空フレーム・後半入力フレームの3つの区間に分ける

            if not iT==1:
                in1_iT=torch.where(
                    torch.rand(size=(T_list[iT],int(datasize/4),int(in_size/2)))<r1, #一様分布でスパイク生成
                    1.0,0.0
                )
                in2_iT=torch.where(
                    torch.rand(size=(T_list[iT],int(datasize/4),int(in_size/2)))<r2, #一様分布でスパイク生成
                    1.0,0.0
                )
            elif iT==1:
                in1_iT=torch.where(
                    torch.rand(size=(T_list[iT],int(datasize/4),int(in_size/2)))<rn, #一様分布でスパイク生成
                    1.0,0.0
                )
                in2_iT=torch.where(
                    torch.rand(size=(T_list[iT],int(datasize/4),int(in_size/2)))<rn, #一様分布でスパイク生成
                    1.0,0.0
                )

            input_spikes.append(torch.cat([in1_iT,in2_iT],dim=2))
        input_spikes=torch.cat(input_spikes,dim=0) #時間方向にstack

        events.append(input_spikes.permute(1,0,2)) #[batch x T x xdim]にpermuteして保持
        targets+=[pattern["out"] for _ in range(input_spikes.shape[1])] #教師をbatch分複製&保持

    events=torch.cat(events,dim=0)
    targets=torch.Tensor(targets)
    print("events: ",events.shape, "targets: ",targets.shape)


    # ランダムに10個のイベントを抽出
    import matplotlib.pyplot as plt
    sample_indices = torch.randperm(events.shape[0])[:10]
    sample_events = events[sample_indices]

    # ヒートマップとして描画
    fig, axes = plt.subplots(10, 1, figsize=(10, 20))
    for i, ax in enumerate(axes):
        ax.imshow(sample_events[i].numpy().T, aspect='auto', cmap='viridis',interpolation='nearest')
        ax.set_title(f'Sample {i+1}/ label {int(targets[sample_indices[i]].item())}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Input Dimension')

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'sample_events_heatmap.png'))
    plt.close()

    
    # Save args to JSON
    args_dict = vars(args)
    args_dict["T-list"]=T_list
    args_dict["rate_pattern"]=rate_pattern
    with open(os.path.join(savedir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)


    indices=torch.arange(events.shape[0],dtype=int)
    indices=torch.randperm(indices.shape[0])  # シャッフル

    train_savedir=savedir/"train"
    test_savedir=savedir/"test"

    if not os.path.exists(train_savedir):
        os.makedirs(train_savedir)
    if not os.path.exists(test_savedir):
        os.makedirs(test_savedir)

    for i,idx in tqdm(enumerate(indices[:args.trainsize]),total=args.trainsize):
        save_to_hdf5(train_savedir/f"data{i}.h5", events[idx], targets[idx])

    for i,idx in tqdm(enumerate(indices[args.trainsize:]),total=args.testsize):
        save_to_hdf5(test_savedir/f"data{i}.h5", events[idx], targets[idx])


if __name__=="__main__":
    main()