import torch
import torch.nn as nn
from snntorch import surrogate
from math import log


class LIF(nn.Module):
    def __init__(self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False,is_train_tau=True):
        """
        :param in_size: currentの入力サイズ
        :param dt: LIFモデルを差分方程式にしたときの⊿t. 元の入力がスパイク時系列ならもとデータと同じ⊿t. 
        :param init_tau: 膜電位時定数τの初期値
        :param threshold: 発火しきい値
        :param vrest: 静止膜電位. 
        :paarm reset_mechanism: 発火後の膜電位のリセット方法の指定
        :param spike_grad: 発火勾配の近似関数
        """
        super(LIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.min_tau=min_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output

        #>> tauを学習可能にするための調整 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 参考 [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        self.is_train_tau=is_train_tau
        init_w=-log(1/(self.init_tau-min_tau)-1)
        if is_train_tau:
            self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # デフォルトの初期化
        elif not is_train_tau:
            self.w=(init_w * torch.ones(size=in_size))  # デフォルトの初期化
        #<< tauを学習可能にするための調整 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.v=0.0
        self.r=1.0 #膜抵抗


    def forward(self,current:torch.Tensor):
        """
        :param current: シナプス電流 [batch x ...]
        """


        # if torch.max(self.tau)<self.dt: #dt/tauが1を超えないようにする
        #     dtau=(self.tau<self.dt)*self.dt
        #     self.tau=self.tau-dtau

        # print(f"tau:{self.tau.shape}, v:{self.v.shape}, current:{current.shape}")
        # print(self.tau)
        # print(self.v)
        # print("--------------")
        device=current.device

        if not self.is_train_tau:
            self.w=self.w.to(device)

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)


        tau=self.min_tau+self.w.sigmoid() #tauが小さくなりすぎるとdt/tauが1を超えてしまう
        dv=self.dt/(tau) * ( -(self.v-self.vrest) + (self.r)*current ) #膜電位vの増分
        self.v=self.v+dv
        spike=self.__fire()
        v_tmp=self.v #リセット前の膜電位
        self.__reset_voltage(spike)

        if not self.output:
            return spike
        else:
            return spike, v_tmp


    def __fire(self):
        v_shift=self.v-self.threshold
        spike=self.spike_grad(v_shift)
        return spike
    

    def __reset_voltage(self,spike):
        if self.reset_mechanism=="zero":
            self.v=self.v*(1-spike.float())
        elif self.reset_mechanism=="subtract":
            self.v=self.v-self.threshold


    def init_voltage(self):
        if not self.v is None:
            self.v=0.0



class DynamicLIF(nn.Module):
    """
    動的にtime constant(TC)が変動するLIF
    """

    def __init__(self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False):
        """
        :param in_size: currentの入力サイズ
        :param dt: LIFモデルを差分方程式にしたときの⊿t. 元の入力がスパイク時系列ならもとデータと同じ⊿t. 
        :param init_tau: 膜電位時定数τの初期値
        :param threshold: 発火しきい値
        :param vrest: 静止膜電位. 
        :paarm reset_mechanism: 発火後の膜電位のリセット方法の指定
        :param spike_grad: 発火勾配の近似関数
        """
        super(DynamicLIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.min_tau=min_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output

        #>> tauを学習可能にするための調整 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 参考 [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        init_w=-log(1/(self.init_tau-min_tau)-1)
        self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # デフォルトの初期化
        #<< tauを学習可能にするための調整 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.v=0.0
        self.r=1.0 #膜抵抗
        self.a=1.0 #タイムスケール


    def forward(self,current:torch.Tensor):
        """
        :param current: シナプス電流 [batch x ...]
        """


        # if torch.max(self.tau)<self.dt: #dt/tauが1を超えないようにする
        #     dtau=(self.tau<self.dt)*self.dt
        #     self.tau=self.tau-dtau

        # #shape debugging
        # try:
        #     print(f"tau:{self.w.shape}, v:{self.v.shape if not self.v==0.0 else 0}, current:{current.shape}")
        #     # print(self.w)
        #     # print(self.v)
        #     print("--------------")
        # except:
        #     pass

        device = current.device  # Get the device of the input tensor

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)

        tau=self.min_tau+self.w.sigmoid() #tauが小さくなりすぎるとdt/tauが1を超えてしまう
        dv=self.dt/(tau*self.a) * ( -(self.v-self.vrest) + (self.a*self.r)*current ) #膜電位vの増分
        self.v=self.v+dv
        spike=self.__fire()
        v_tmp=self.v #リセット前の膜電位
        self.__reset_voltage(spike)

        if not self.output:
            return spike
        else:
            return spike, v_tmp


    def __fire(self):
        v_shift=self.v-self.threshold
        spike=self.spike_grad(v_shift)
        return spike
    

    def __reset_voltage(self,spike):
        if self.reset_mechanism=="zero":
            self.v=self.v*(1-spike.float())
        elif self.reset_mechanism=="subtract":
            self.v=self.v-self.threshold


    def init_voltage(self):
        self.v=0.0
