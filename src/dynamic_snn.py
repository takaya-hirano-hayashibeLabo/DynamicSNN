import torch
import torch.nn as nn
from snntorch import surrogate



class DynamicLIF(nn.Module):
    """
    動的にtime constant(TC)が変動するLIF
    """

    def __init__(self,dt,init_tau=0.5,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False):
        """
        :param dt: LIFモデルを差分方程式にしたときの⊿t. 元の入力がスパイク時系列ならのとデータと同じ⊿t. 
        :param init_tau: 膜電位時定数τの初期値
        :param threshold: 発火しきい値
        :param vrest: 静止膜電位. 計算がややこしくなるので0でいい気がする
        :paarm reset_mechanism: 発火後の膜電位のリセット方法の指定
        :param spike_grad: 発火勾配の近似関数
        """
        super(DynamicLIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output

        self.is_init_tau=False
        self.tau=None
        self.v=None
        self.r=1.0 #膜抵抗

        self.tau_pool=None #動的に動かすときに, 学習済みのtauをpoolしておく用のプロパティ


    def forward(self,current:torch.Tensor):
        """
        :param current: シナプス電流 [batch x ...]
        """

        if not self.is_init_tau: #tauの形状はデータが流れてくるまでわからない
            self.__init_internal_state(current)
            self.is_init_tau=True

        # if torch.max(self.tau)<self.dt: #dt/tauが1を超えないようにする
        #     dtau=(self.tau<self.dt)*self.dt
        #     self.tau=self.tau-dtau

        dv=self.dt/self.tau * ( -(self.v-self.vrest) + self.r*current ) #膜電位vの増分
        self.v=self.v+dv
        spike=self.__fire()
        v_tmp=self.v #リセット前の膜電位
        self.__reset_voltage(spike)

        if not self.output:
            return spike
        else:
            return spike, v_tmp


    def __init_internal_state(self,x:torch.Tensor):
        """
        入力の形状に応じて, tauとvを初期化する関数
        :param x: シナプス電流 [batch x ...]
        """

        state_shape=x.shape[1:] #バッチより後ろのサイズを取る
        device=x.device
        if self.tau is None:
            self.tau = nn.Parameter(self.init_tau*torch.ones(state_shape)).to(device) # 学習可能なパラメータとしてtauをセット
        if self.v is None:
            self.v=torch.zeros(state_shape).to(device)


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

        if self.is_init_tau:
            self.v=self.v*0.0



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

        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()


        modules=[]

        #>> 入力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.in_size, self.hiddens[0]),
            DynamicLIF(
                dt=self.dt,init_tau=self.init_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False
            ),
            nn.Dropout(self.dropout),
        ]
        #<< 入力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> 中間層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        prev_hidden=self.hiddens[0]
        for hidden in self.hiddens[1:]:
            modules+=[
                nn.Linear(prev_hidden, hidden),
                DynamicLIF(
                    dt=self.dt,init_tau=self.init_tau,threshold=self.v_threshold,vrest=self.v_rest,
                    reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False
                ),
                nn.Dropout(self.dropout),
            ]
            prev_hidden=hidden
        #<< 中間層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> 出力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.hiddens[-1], self.out_size),
            DynamicLIF(
                dt=self.dt,init_tau=self.init_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True
            ),
        ]
        #<< 出力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.net=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.net:
            if isinstance(layer,DynamicLIF):
                layer.init_voltage()


    def __set_dynamic_params(self,a):
        """
        LIFの時定数&膜抵抗を変動させる関数
        :param a: [スカラ]その瞬間の速度倍率
        """
        for layer in self.net:
            if isinstance(layer,DynamicLIF): #ラプラス変換によると速度倍率をかけると上手く行くはず

                if layer.tau_pool is None:
                    layer.tau_pool=layer.tau.clone() #学習済みのtauをpoolしとく
                layer.tau = layer.tau_pool.clone()*a
                layer.r=1.0*a


    def forward(self,s:torch.Tensor):
        """
        :param s: スパイク列 [T x batch x ...]
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """

        T=s.shape[0]
        self.__init_lif()

        out_s,out_v=[],[]
        for t in range(T):
            st,vt=self.net(s[t])
            out_s.append(st)
            out_v.append(vt)

        out_s=torch.stack(out_s,dim=0)
        out_v=torch.stack(out_v,dim=0)

        return out_s,out_v


    def dynamic_forward(self,s:torch.Tensor,a:torch.Tensor):
        """
        :param s: スパイク列 [T x batch x ...]
        :param a: 速度倍率リスト [T] バッチ間で速度倍率は統一する
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """
        self.net.eval() #絶対学習しない

        T=s.shape[0]
        self.__init_lif()

        out_s,out_v=[],[]
        for t in range(T):

            with torch.no_grad():
                self.__set_dynamic_params(a[t])
                st,vt=self.net(s[t])

            out_s.append(st)
            out_v.append(vt)

        out_s=torch.stack(out_s,dim=0)
        out_v=torch.stack(out_v,dim=0)

        return out_s,out_v
    


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