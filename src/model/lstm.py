import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self,conf):
        super(LSTM,self).__init__()

        self.in_size = conf['in-size']
        self.hidden=conf["hidden"]
        self.hidden_num=conf["hidden-num"]
        self.out_size=conf["out-size"]


        self.lstm=nn.LSTM(
            input_size=self.in_size,
            hidden_size=self.hidden,
            num_layers=self.hidden_num,
        )


        #>> 出力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules=[]
        modules+=[
            nn.Linear(self.hidden,self.out_size),
        ]
        self.out_layer=nn.Sequential(*modules)
        #<< 出力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    def forward(self,x:torch.Tensor):
        """
        param: x: [time-sequence x batch x xdim]
        return out: [batch x outdim]
        """

        out, (hn,cn)=self.lstm(x)
        out=self.out_layer(out[-1]) #最終stepだけ出力とする

        return out