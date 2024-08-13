from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=True,dropout=0.3):
        """
        CNNのみの残差ブロック
        このResidualでは入出力のサイズは変わらない (channelはもちろん変わる)
        :param num_block: 入力のCNN以外に何個のCNNを積むか
        """
        super(ResidualBlock,self).__init__()

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
                nn.ReLU(inplace=False)
            ]
        for i in range(num_block):
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    ),
                    nn.ReLU(inplace=False)
                ]
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )

    def forward(self,x):
        """
        残差を足して出力
        """
        out=self.model(x)
        residual=self.shortcut(x)
        return out+residual
