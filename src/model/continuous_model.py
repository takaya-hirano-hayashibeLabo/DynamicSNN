"""
時系列分散表現から連続値を予測するためのクラス
ここをattentionにしてもいいと思う(めんどいからやらない)
"""

import torch
import torch.nn as nn
from snntorch import surrogate
from math import log
from tqdm import tqdm
import itertools

from .residual_block import ResidualBlock, ResidualDynaLIFBlock
from .scale_predictor import ScalePredictor
from .lif_model import DynamicLIF
from .dynamic_snn import DynamicSNN


class ContinuousSNN(nn.Module):

    def __init__(self,conf:dict, time_encoder:DynamicSNN):
        """
        :param conf: 連続値予測の出力層のconfig
        :param time_encoder: 時系列表現に落とすモデル(DynamicSNNとか)
        """
        super(ContinuousSNN,self).__init__()

        self.time_encoder=time_encoder

        self.in_size=conf["in-size"]
        self.out_size=conf["out-size"]
        self.hiddens=conf["hiddens"]
        self.dropout=conf["dropout"]
        self.clip_norm=conf["clip-norm"]


        modules=[]
        modules+=[
            nn.Conv1d(in_channels=self.in_size, out_channels=self.hiddens[0],kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        
        prev_hidden=self.hiddens[0]
        for i in range(len(self.hiddens)-1):
            hidden=self.hiddens[i+1]
            modules+=[
                nn.Conv1d(in_channels=prev_hidden,out_channels=hidden,kernel_size=1),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]

        modules+=[
            nn.Conv1d(in_channels=self.hiddens[-1],out_channels=self.out_size,kernel_size=1),
            nn.Tanh() #出力は-1~1
        ]

        self.model=nn.Sequential(*modules)


    def clip_gradients(self):
        """
        Clips gradients to prevent exploding gradients.
        
        :param max_norm: Maximum norm of the gradients.
        """
        # Clip gradients for the ContiuousModel parameters
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        

    def forward(self,inspikes:torch.Tensor):
        """
        :param inspikes: [N x T x xdim]
        """

        in_sp=inspikes.permute((1,0,*[i+2 for i in range(inspikes.ndim-2)])) #[T x N x xdim]
        _,_,out_v=self.time_encoder.forward(in_sp) #[T x N x outdim]

        out_v=out_v.permute(1,2,0) #[N x outdim x T] 時間方向は関連させない
        out:torch.Tensor=self.model(out_v) #[N x outdim x T]

        return out.permute(0,2,1) #[N x T x outdim]