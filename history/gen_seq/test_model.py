import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(ROOT))

import argparse
from pathlib import Path
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import load_yaml
from src.model import DynamicSNN
from encoder import encode2spike
from train_genseq import create_windows, ft


def plot_and_save_inseq(t,in_seq, f_next, n, saveto, t_head):
    """
    in_seqのバッチうち, n個の要素の時系列を描画して保存する関数
    f_nextを教師データとして描画
    """
    T_seq = in_seq.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n))
    axes=[axes] if n==1 else axes
    for i in range(n):
        axes[i].axvspan(0, t[t_head], color='red', alpha=0.2)  # t_headまで背景を薄い赤で塗る
        axes[i].plot(t, in_seq[i].cpu().numpy(), label="input")
        if f_next is not None:
            axes[i].plot(t, f_next[i].cpu().numpy(), linestyle='--',alpha=0.5)
        axes[i].set_title(f"Time Series for Batch Element {i}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(saveto)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help="configのあるパス", default=Path(__file__).parent)
    parser.add_argument("--resultpath",default="result")
    parser.add_argument("--device", default=0, help="GPUの番号")
    parser.add_argument("--modelname",default="final")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    resultpath = Path(args.target) / args.resultpath
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf = load_yaml(Path(args.target) / "conf.yml")
    model_conf = conf["model"]

    # モデルの準備
    if model_conf["type"] == "dynamic-snn".casefold():
        model = DynamicSNN(model_conf)
    else:
        raise ValueError(f"model type {model_conf['type']} is not supported...")
    model.load_state_dict(torch.load(resultpath / f"model_{args.modelname}.pth"))
    model.to(device)
    model.eval()

    outnet = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=model_conf["out-size"], out_channels=model_conf["out-size"], kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=model_conf["out-size"], out_channels=1, kernel_size=1),
        torch.nn.Tanh()
    )
    outnet.load_state_dict(torch.load(resultpath / f"outnet_{args.modelname}.pth"))
    outnet.to(device)
    outnet.eval()

    # テストデータの準備
    sequence_length = 2000
    t=np.linspace(0,30,sequence_length)
    f = torch.tensor(ft(t)).unsqueeze(-1)

    window = 400
    overlap = 0.8
    test_in = create_windows(f, window, overlap)  # [N x T x m]

    n_head=25
    n_t=window-n_head
    threshold=np.linspace(-1,1,model_conf["in-size"])
    in_seq=test_in[:,:n_head].detach().clone() #[N x T x m]

    scale=2.0
    with torch.no_grad():
        for n in tqdm(range(n_t)):

            in_spikes=encode2spike(in_seq,threshold) #[N x n_head x c x m]
            in_spikes=in_spikes.squeeze()
            in_spikes=in_spikes.permute(1,0,2) #[T x N x cm]
  
            _,_,out_v=model.dynamic_forward_genseq(in_spikes.to(device),a=1.0,head_idx=n_head) #dynasnnを使わない
            # _,_,out_v=model.dynamic_forward_genseq(in_spikes.to(device),a=scale,head_idx=n_head) #[T x N x h]
            out_v=out_v.permute(1,2,0)
            out:torch.Tensor=outnet(out_v) #差分予測 [N x m x T]

            # print(out[:,:,-1].flatten())

            f_next_=in_seq[:,-1].to(device)+out[:,:,-1]/scale #差分の足し合わせ [N x m]
            f_next_=f_next_.unsqueeze(1)
            f_next_[(f_next_>1)]=1
            f_next_[(f_next_<-1)]=-1
            # print("fnext_",f_next_.shape)

            in_seq=torch.cat([in_seq.to(device),f_next_],dim=1) #時間方向にcat
            # print("in_seq: ",in_seq.shape)

    plot_and_save_inseq(t[:window], in_seq,create_windows(f,window,overlap),n=5,saveto=resultpath/f"regression_scale{scale:.2f}.png",t_head=n_head)


if __name__ == "__main__":
    main()