import torch
import numpy as np
from math import log,exp


class ScalePredictor():
    """
    入力からタイムスケールを予測するクラス
    """
    def __init__(self,datatype="xor"):

        self.datatype=datatype
        self.data_trj=torch.Tensor(np.array([]))


    def predict_scale(self,data:torch.Tensor):
        """
        1ステップ分のデータを入力とする
        :param data: [batch x x_dim]
        """

        scale=1
        if self.datatype=="xor":
            scale=self.__predict_xor(data)
        if self.datatype=="gesture":
            scale=self.__predict_gesture(data)

        return scale


    def __predict_xor(self,data:torch.Tensor):
        """
        xorの1ステップ分のデータが入る
        :param data: [batch x xdim]
        """

        #>> scaleとfiring rateで線形回帰したときの係数とwindow >>>>>>>>
        window_size=120
        slope,intercept=-1.0437631068421338,-0.6790105922709921
        #<< scaleとfiring rateで線形回帰したときの係数とwindow <<<<<<<<
        
        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>window_size:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-window_size):] #長くなりすぎたらカット
        
        scale_log=log(self.firing_rate+1e-10)*slope + intercept
        scale=exp(scale_log)

        return scale
    

    def __predict_gesture(self,data:torch.Tensor):
        """
        テスト済み. 想定した通りの挙動をしている
        gestureの1ステップ分のデータが入る
        :param data: [batch x channel x w x h]
        """

        #>> 線形回帰したときのパラメータ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        window_size=100
        slope=-1.3015304644834496
        intercept=-4.641658840504729
        #<< 線形回帰したときのパラメータ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>window_size:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-window_size):] #長くなりすぎたらカット
        
        scale_log=log(self.firing_rate+1e-10)*slope + intercept
        scale=exp(scale_log)

        return scale

    def reset_trj(self):
        self.data_trj=torch.Tensor(np.array([]))
    
    @property
    def firing_rate(self):
        fr=torch.mean(self.data_trj,dim=1) #時間方向に平均
        fr=torch.mean(fr,dim=tuple([i+1 for i in range(fr.ndim-1)])) #空間方向に平均
        fr=torch.mean(fr) #バッチ方向に平均
        return fr.item()