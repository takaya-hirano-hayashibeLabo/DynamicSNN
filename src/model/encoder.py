"""
スパイクエンコーダ
速度変化に応じてガウス関数のタイムスケールが上手く起きるようにしたもの
"""
import torch

class IFEncoder():
    """
    積分発火(Integrated Firing)のエンコーダ
    イベントデータに対してこれを噛ませることで、単純なタイムスケーリングになりそうな感じがする
    入力イベントデータは1にクリップしてないものを用いる
    """
    def __init__(self,threshold,device="cpu"):
        self.threshold=threshold #発火の閾値
        self.device=device
        self.skip_ndim=2

    def __call__(self,x:torch.Tensor):
        """
        イベントデータは1クリップしてないものを入力する
        :param x: [N x T x xdim]
        :return out_spikes: [N x T x xdim]
        """
        if not isinstance(x,torch.Tensor):
            x=torch.Tensor(x).to(self.device)

        N,T=x.shape[0],x.shape[1]
        internal_state=torch.zeros(size=(N,)+self._get_xdim(x)).to(self.device)
        out_spikes=torch.zeros_like(x).to(self.device)
        for t in range(T):
            internal_state+=x[:,t]
            out_spikes[:,t]=torch.where(self._is_over_threshold(internal_state),1.0,0.0).to(self.device)
            internal_state=self._reset_internal_state(internal_state)
        return out_spikes
    
    def _get_xdim(self,x:torch.Tensor):
        xdim_size=tuple(
            x.shape[i+self.skip_ndim] for i in range(x.ndim-self.skip_ndim)
        ) #2次元目以降の次元サイズ
        return xdim_size
            
    def _is_over_threshold(self,internal_state):
        return internal_state>=self.threshold

    def _reset_internal_state(self,internal_state:torch.Tensor):
        new_internal_state=internal_state.clone().detach().to(self.device)
        new_internal_state[self._is_over_threshold(internal_state)]-=self.threshold
        return new_internal_state


class ThresholdEncoder():
    """
    閾値を超えたらスパイクを出力する
    """
    def __init__(self,thr_max,thr_min,resolution:int,device="cpu"):
        """
        :param thr_max, thr_min: 閾値の最大, 最小
        :param resolution: 閾値の分割数
        """
        self.threshold=torch.linspace(thr_min,thr_max,resolution).to(device)
        self.resolution=resolution
        self.skip_ndim=2 #0,1次元目は飛ばす(N x Tとわかっているため)
        self.device=device

    def _is_same_ndim(self,x:torch.Tensor):
        """
        入力とthresholdの次元数が一致しているか
        """
        return x.ndim==self.threshold.ndim
    
    def _reshape_threshold(self,x:torch.Tensor):
        """
        :param x: [N x T x xdim]
        :return: [N x T x resolution x xdim]
        """
        nxdim=x.ndim-self.skip_ndim #2次元目以降の次元数
        new_thr_size=(1,1,self.resolution)+tuple(1 for _ in range(nxdim))
        
        self.threshold=self.threshold.view(*new_thr_size)

        # xdim_size=tuple(
        #     [x.shape[i+skip_ndim] for i in range(x.ndim-skip_ndim)]
        # ) #2次元目以降の次元サイズ

    def _get_xdim(self,x:torch.Tensor):
        xdim_size=tuple(
            x.shape[i+self.skip_ndim] for i in range(x.ndim-self.skip_ndim)
        ) #2次元目以降の次元サイズ
        return xdim_size
    
    def __call__(self,x:torch.Tensor):
        """
        :param x: [N x T x xdim]
        :return: [N x T x resolution x xdim]
        """
        if not isinstance(x, torch.Tensor):
            x=torch.Tensor(x)

        N, T = x.shape[0],x.shape[1]

        out_spike_shape=(N,T,self.resolution) + self._get_xdim(x)
        out_spike=torch.zeros(out_spike_shape,device=self.device)

        #>> 入力に合わせてthresholdの次元数を調整 >>
        if not self._is_same_ndim(x): self._reshape_threshold(x)

        #>> 入力をシフトして前後の値を比較 >>
        x_prev=x[:,:-1].unsqueeze(2) #unsqueezeはthreshold次元を追加
        x_next=x[:,1:].unsqueeze(2)

        # >> 閾値をまたいだらTrueとする >>
        spike_mask=(
            ((x_prev<=self.threshold) & (self.threshold<x_next)) |
            ((x_next<self.threshold) & (self.threshold<=x_prev))
        )

        out_spike[:,1:]=spike_mask.float().to(self.device)

        return out_spike
        

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
