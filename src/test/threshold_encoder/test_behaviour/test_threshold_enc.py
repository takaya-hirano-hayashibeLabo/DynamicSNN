"""
テスト項目
・入出力次元のみ
=単純な動作確認のみ

未検証項目
・本当にタイムスケールの変換になるか？
  (プログラムの動作というよりは, 原理的に理想状態に近づくか？の部分)
"""

import yaml 
from pathlib import Path
import sys
MODELDIR=Path(__file__).parent.parent.parent.parent
sys.path.append(str(MODELDIR))
import torch
import matplotlib.pyplot as plt

from model.encoder import ThresholdEncoder

def save_spike_frames_histogram(spike_frames, filename="spike_frames_histogram.png"):
    plt.hist(spike_frames.flatten().cpu().numpy(), bins=50)
    plt.title('Spike Frames Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()

def main():

    # Create a random input tensor
    T = 30  # Number of timesteps
    batch_size = 10
    c,h,w=2,8,8
    spike_rate=0.3

    # >> イベント動画と同じ形式 >>
    input_spikes=torch.where(
        torch.rand(batch_size,T,c,h,w)<spike_rate,
        1.0,0.0
    )
    spike_frames=1.5*input_spikes[:,:,0]+0.5*input_spikes[:,:,1]-1 #連続値へ変換
    # save_spike_frames_histogram(spike_frames)
    skip_dim=2
    print("frame ndim:",spike_frames.ndim, "xdim:",[spike_frames.shape[i+skip_dim] for i in range(spike_frames.ndim-skip_dim)])
    # >> イベント動画と同じ形式 >>


    #>> encoderの使い方はこれだけ >>
    thr_enc=ThresholdEncoder(thr_max=0.75,thr_min=-0.75,resolution=3)
    out=thr_enc.__call__(spike_frames) #out: [N x T x resolution x h x w]
    #>> encoderの使い方はこれだけ >>

    print(thr_enc.threshold)
    print(thr_enc.threshold.shape)

    print("out shape: ",out.shape)

if __name__=="__main__":
    main()