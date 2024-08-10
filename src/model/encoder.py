"""
スパイクエンコーダ
速度変化に応じてガウス関数のタイムスケールが上手く起きるようにしたもの
"""
import torch

class DiffEncoder():
    """
    時間変量に応じてスパイクが出力される
    こうすることで、タイムスケーリングに対して理想に近いスパイクがでるはず
    """

    def __init__(self,threshold):
        self.threshold=threshold

        self.is_init_state=False
        self.prev_x=torch.zeros(1)
        self.state=torch.zeros(1)

    def step_forward(self,xt:torch.Tensor):
        """
        :param xt: [batch x xdim]. 1ステップの入力
        :retrun out_spike: [batch x xdim] 出力スパイク
        """

        with torch.no_grad():
            if not self.is_init_state:
                self.prev_x=xt.clone()
                self.state=torch.zeros_like(xt)
                self.is_init_state=True
            
            dx=torch.abs(xt-self.prev_x) #1stepの絶対変量をとる. 本来はローパスみたいなのをかけたほうがいい
            self.state+=dx
            out_spike=torch.where(self.state>self.threshold,1.0,0.0) #変量がしきい値を超えたら発火
            self.state[self.state>self.threshold]=0.0 #発火したらstateを0に戻す
            self.prev_x=xt

        return out_spike
    

    def reset_state(self):
        """
        1シーケンスが終わったときにリセットする
        """
        self.is_init_state=False
        self.state =torch.zeros(1)
        self.prev_x=torch.zeros(1)



from .snn import LIF
from torch import nn
from snntorch import surrogate

class DirectCSNNEncoder(nn.Module):
    """
    連続値を受け取ってスパイクを返すEncoder
    """

    def __init__(self,conf:dict):
        super(DirectCSNNEncoder,self).__init__()

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.pool_type = conf["pool-type"]
        
        self.output_mem=conf["output-membrane"]
        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.is_train_tau=conf["train-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()


        modules=[]

        modules+=[
            nn.Conv2d(
                in_channels=self.in_channel,out_channels=self.in_channel,
                kernel_size=3,stride=1,padding=1,bias=True
            ),
            LIF(
                in_size=(self.in_channel,self.in_size,self.in_size), dt=self.dt,
                init_tau=self.init_tau, min_tau=self.min_tau,
                threshold=self.v_threshold, vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism, spike_grad=self.spike_grad,
                output=False,is_train_tau=self.is_train_tau
            )
        ]

        self.model=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.model:
            if isinstance(layer,LIF):
                layer.init_voltage()

    def reset_state(self):
        self.__init_lif()

    def step_forward(self,x:torch.Tensor):
        """
        :param x: [batch x xdim...]
        """
        return self.model(x)
