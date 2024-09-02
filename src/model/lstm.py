import torch.nn as nn
import torch

from .residual_block import ResidualBlock

class LSTM(nn.Module):

    def __init__(self,conf):
        super(LSTM,self).__init__()

        self.in_size = conf['in-size']
        self.hidden=conf["hidden"]
        self.hidden_num=conf["hidden-num"]
        self.out_size=conf["out-size"]
        self.clip_norm=conf["clip-norm"] if "clip-norm" in conf.keys() else 1.0


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
    
    def clip_gradients(self):
        """
        Clips gradients to prevent exploding gradients.
        
        :param max_norm: Maximum norm of the gradients.
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape



def add_residual_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,residual_block_num,is_bn,pool_type,pool_size,dropout,
        ):
    """
    param: in_size: 幅と高さ (正方形とする)
    param: in_channel: channel size
    param: out_channel: 出力チャネルのサイズ
    param: kernel: カーネルサイズ
    param: stride: ストライドのサイズ
    param: padding: パディングのサイズ
    param: is_bias: バイアスを使用するかどうか
    param: residual_block_num: ResBlock内のCNNの数 (0でもいい)
    param: is_bn: バッチ正規化を使用するかどうか
    param: pool_type: プーリングの種類 ("avg"または"max")
    param: pool_size: プールのサイズ
    param: dropout: dropout rate
    """
    
    block=[]
    block.append(
        ResidualBlock(
            in_channel=in_channel,out_channel=out_channel,
            kernel=kernel,stride=stride,padding=padding,
            num_block=residual_block_num,bias=is_bias
        )
    )

    if is_bn:
        block.append(
            nn.BatchNorm2d(out_channel)
        )

    if pool_size>0:
        if pool_type=="avg".casefold():
            block.append(nn.AvgPool2d(pool_size))
        elif pool_type=="max".casefold():
            block.append(nn.MaxPool2d(pool_size))

    #blockの出力サイズを計算
    block_outsize=get_conv_outsize(nn.Sequential(*block),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]

    block.append(
        nn.ReLU()
    )

    if dropout>0:
        block.append(nn.Dropout2d(dropout, inplace=False))
    
    return block, block_outsize




class ResNetLSTM(nn.Module):

    def __init__(self,conf):
        super(ResNetLSTM,self).__init__()

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.residual_blocks=conf["residual-block"] #残差ブロックごとのCNNの数
        self.pool_type = conf["pool-type"]
        self.pool_size=conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]

        self.lstm_h_num=conf["lstm-hidden-num"]
        self.lstm_hidden=conf["lstm-hidden"]



        #>> 入力層 1次元にflattenする用のlayer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules=[]

        in_c=self.in_channel
        in_size=self.in_size
        for i, h_c in enumerate(self.hiddens):

            block,block_outsize=add_residual_block(
                in_size=in_size,in_channel=in_c,out_channel=h_c,
                kernel=3,stride=1,padding=1,is_bias=True,
                residual_block_num=self.residual_blocks[i],is_bn=self.is_bn,
                pool_type=self.pool_type,pool_size=self.pool_size[i],dropout=self.dropout
            )
            modules+=block
            in_c=h_c
            in_size=block_outsize[-1]

        modules+=[
            nn.Flatten(),
            nn.Linear(block_outsize[1]*block_outsize[2]*block_outsize[3],self.linear_hidden,bias=True),
            nn.ReLU(),
        ]

        self.in_layer=nn.Sequential(*modules)
        #<< 入力層 1次元にflattenする用のlayer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        self.lstm=nn.LSTM(
            input_size=self.linear_hidden,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_h_num,
        )


        #>> 出力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules=[]
        modules+=[
            nn.Linear(self.lstm_hidden,self.out_size),
        ]
        self.out_layer=nn.Sequential(*modules)
        #<< 出力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    def forward(self,x:torch.Tensor):
        """
        param: x: [time-sequence x batch x xdim]
        return out: [batch x outdim]
        """

        T,batch_size,c,h,w=x.shape
        out=self.in_layer(x.reshape(T*batch_size,c,h,w))
        out=out.reshape(T, batch_size, -1)
        out, _=self.lstm(out)
        out=self.out_layer(out[-1]) #最終stepだけ出力とする

        return out
