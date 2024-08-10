import torch
import torch.nn as nn
from snntorch import surrogate
from math import log
from tqdm import tqdm

from .scale_predictor import ScalePredictor

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
        :param vrest: 静止膜電位. 計算がややこしくなるので0でいい気がする
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
        if not self.v is None:
            self.v=0.0



class DynamicSNN(nn.Module):
    """
    time constant(TC)が動的に変動するSNN
    """

    def __init__(self,conf:dict):
        super(DynamicSNN,self).__init__()

        self.in_size=conf["in-size"]
        self.hiddens = conf["hiddens"]
        self.out_size = conf["out-size"]
        self.dropout=conf["dropout"]
        self.output_mem=conf["output-membrane"]

        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()


        modules=[]
        is_bias=False #biasはつけちゃダメ. ラプラス変換の式が成り立たなくなる.

        #>> 入力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.in_size, self.hiddens[0],bias=is_bias),
            DynamicLIF(
                in_size=(self.hiddens[0],),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False
            ),
            nn.Dropout(self.dropout),
        ]
        #<< 入力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> 中間層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        prev_hidden=self.hiddens[0]
        for hidden in self.hiddens[1:]:
            modules+=[
                nn.Linear(prev_hidden, hidden,bias=is_bias),
                DynamicLIF(
                    in_size=(hidden,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                    reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False
                ),
                nn.Dropout(self.dropout),
            ]
            prev_hidden=hidden
        #<< 中間層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> 出力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.hiddens[-1], self.out_size,bias=is_bias),
            DynamicLIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True
            ),
        ]
        #<< 出力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.model=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                layer.init_voltage()


    def __set_dynamic_params(self,a):
        """
        LIFの時定数&膜抵抗を変動させる関数
        :param a: [スカラ]その瞬間の時間スケール
        """
        for layer in self.model:
            if isinstance(layer,DynamicLIF): #ラプラス変換によると時間スケールをかけると上手く行くはず
                layer.a = a


    def __reset_params(self):
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                layer.a=1.0



    def forward(self,s:torch.Tensor):
        """
        :param s: スパイク列 [T x batch x ...]
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """

        T=s.shape[0]
        batch_size=s.shape[1]
        self.__init_lif()

        # Preallocate output tensors for better performance
        out_s = torch.empty(T, batch_size, self.out_size, device=s.device)
        out_v = torch.empty(T, batch_size, self.out_size, device=s.device)
        for t in range(T):
            st,vt=self.model(s[t])
            out_s[t]=st
            out_v[t]=vt

        if self.output_mem:
            return out_s,out_v
        
        elif not self.output_mem:
            return out_s


    def dynamic_forward(self,s:torch.Tensor, scale_predictor:ScalePredictor):
        """
        時間スケールが未知のときのdynamic_forward
        :param s: スパイク列 [T x batch x ...]
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """
        self.model.eval() #絶対学習しない

        T=s.shape[0]
        self.__init_lif()

        out_s,out_v=[],[]
        for t in tqdm(range(T)):

            with torch.no_grad():
                a=scale_predictor.predict_scale(s[t]) #現在のscaleを予測
                self.__set_dynamic_params(a)
                st,vt=self.model(s[t])

            out_s.append(st)
            out_v.append(vt)

        out_s=torch.stack(out_s,dim=0)
        out_v=torch.stack(out_v,dim=0)

        self.__reset_params()
        scale_predictor.reset_trj()

        if self.output_mem:
            return out_s,out_v
        
        elif not self.output_mem:
            return out_s



    def dynamic_forward_v1(self,s:torch.Tensor,a:torch.Tensor):
        """
        時間スケールが既知のときのdynamic_forward
        :param s: スパイク列 [T x batch x ...]
        :param a: 時間スケールリスト [T] バッチ間で時間スケールは統一する
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """
        self.model.eval() #絶対学習しない

        T=s.shape[0]
        self.__init_lif()

        out_s,out_v=[],[]
        for t in range(T):

            with torch.no_grad():
                self.__set_dynamic_params(a[t])
                st,vt=self.model(s[t])

            out_s.append(st)
            out_v.append(vt)

        out_s=torch.stack(out_s,dim=0)
        out_v=torch.stack(out_v,dim=0)

        self.__reset_params()

        if self.output_mem:
            return out_s,out_v
        
        elif not self.output_mem:
            return out_s


def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape



def add_csnn_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,is_bn,pool_type,dropout,
        lif_dt,lif_init_tau,lif_min_tau,lif_threshold,lif_vrest,lif_reset_mechanism,lif_spike_grad,lif_output
        ):
    """
    param: in_size: 幅と高さ (正方形とする)
    param: in_channel: channel size
    param: out_channel: 出力チャネルのサイズ
    param: kernel: カーネルサイズ
    param: stride: ストライドのサイズ
    param: padding: パディングのサイズ
    param: is_bias: バイアスを使用するかどうか
    param: is_bn: バッチ正規化を使用するかどうか
    param: pool_type: プーリングの種類 ("avg"または"max")
    param: dropout: dropout rate
    param: lif_dt: LIFモデルの時間刻み
    param: lif_init_tau: LIFの初期時定数
    param: lif_min_tau: LIFの最小時定数
    param: lif_threshold: LIFの発火しきい値
    param: lif_vrest: LIFの静止膜電位
    param: lif_reset_mechanism: LIFの膜電位リセットメカニズム
    param: lif_spike_grad: LIFのスパイク勾配関数
    param: lif_output: LIFの出力を返すかどうか
    """
    
    block=[]
    block.append(
        nn.Conv2d(
            in_channels=in_channel,out_channels=out_channel,
            kernel_size=kernel,stride=stride,padding=padding,bias=is_bias
        )
    )

    if is_bn:
        block.append(
            nn.BatchNorm2d(out_channel)
        )

    if pool_type=="avg".casefold():
        block.append(nn.AvgPool2d(2))
    elif pool_type=="max".casefold():
        block.append(nn.MaxPool2d(2))

    #blockの出力サイズを計算
    block_outsize=get_conv_outsize(nn.Sequential(*block),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]

    block.append(
        DynamicLIF(
            in_size=tuple(block_outsize[1:]), dt=lif_dt,
            init_tau=lif_init_tau, min_tau=lif_min_tau,
            threshold=lif_threshold, vrest=lif_vrest,
            reset_mechanism=lif_reset_mechanism, spike_grad=lif_spike_grad,
            output=lif_output
        )
    )

    if dropout>0:
        block.append(nn.Dropout2d(dropout))
    
    return block, block_outsize


class DynamicCSNN(DynamicSNN):
    """
    DynamicSNNのCNNバージョン 
    CNNでは1層スタックするごとにサイズが1/2になる
    """

    def __init__(self,conf):
        super(DynamicCSNN,self).__init__(conf)

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.pool_type = conf["pool-type"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]
        
        self.output_mem=conf["output-membrane"]
        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()



        modules=[]

        #>> 畳み込み層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for hidden_c in self.hiddens:

            block,block_outsize=add_csnn_block(
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # カーネルサイズ、ストライド、パディングの設定
                is_bias=False, is_bn=self.is_bn, pool_type=self.pool_type, dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False  # 出力を返さない設定
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< 畳み込み層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> 線形層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Flatten(),
            nn.Linear(block_outsize[1]*block_outsize[2]*block_outsize[3],self.linear_hidden,bias=False),
            DynamicLIF(
                in_size=(self.linear_hidden,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False
            ),
            nn.Linear(self.linear_hidden,self.out_size,bias=False),
            DynamicLIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True
            ),
        ]
        #<< 線形層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)



    

if __name__=="__main__":
    """
    テスト項目
    :forwad
        :モデルの入出力
        :GPUの利用
        :DynamicLIFのτのサイズ
        :モデルの保存とロード

    :dynamic_forward
        :モデルの入出力
        :GPUの利用
        :DynamicLIFのτのサイズ
        :DynamicLIFのτの変動
    """

    import yaml
    from pathlib import Path

    # Load configuration
    with open(Path(__file__).parent/"test/conf.yml", "r") as file:
        conf = yaml.safe_load(file)

    # Create a random input tensor
    T = 1000  # Number of timesteps
    batch_size = 5
    input_size = conf["model"]["in-size"]
    device = "cuda:0"

    input_data = torch.randn(T, batch_size, input_size).to(device)

    # Initialize the model
    model = DynamicSNN(conf["model"])
    model.to(device)
    print(model)


    print("test 'forward' func"+"-"*100)
    # Forward pass
    out_s, out_v = model(input_data)

    # Print the shapes of the outputs
    print("Output spikes shape:", out_s.shape)
    print("Output voltages shape:", out_v.shape)

    # Print the size of tau for each DynamicLIF layer
    for layer in model.net:
        if isinstance(layer, DynamicLIF):
            print("Tau size:", layer.tau.size())

    # Save the model
    model_path = Path(__file__).parent / "test/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Load the model
    loaded_model = DynamicSNN(conf["model"])
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    print("Model loaded successfully")

    # Verify the loaded model
    out_s_loaded, out_v_loaded = loaded_model(input_data)
    print("Output spikes shape (loaded model):", out_s_loaded.shape)
    print("Output voltages shape (loaded model):", out_v_loaded.shape)

    # Print the size of tau for each DynamicLIF layer in the loaded model
    for layer in loaded_model.net:
        if isinstance(layer, DynamicLIF):
            print("Tau size (loaded model):", layer.tau.size())



    print("\ntest 'dynamic_forward' func"+"-"*100)
    # Dynamic forward pass
    a = torch.rand(T).to(device)  # Random speed multipliers
    out_s_dyn, out_v_dyn = model.dynamic_forward(input_data, a)

    # Print the shapes of the dynamic outputs
    print("Dynamic output spikes shape:", out_s_dyn.shape)
    print("Dynamic output voltages shape:", out_v_dyn.shape)

    # Print the size of tau for each DynamicLIF layer after dynamic forward
    for layer in model.net:
        if isinstance(layer, DynamicLIF):
            print("Tau size (dynamic forward):", layer.tau.size())