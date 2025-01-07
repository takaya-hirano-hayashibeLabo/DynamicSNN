from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
import seaborn as sns
sns.set(style="darkgrid")

def cm2inch(value):
    return value/2.54


def get_ticks_value(ax:plt.Axes,width:int,bias:float,axis:str="x"):
    """
    メモリの軸の数値の設定
    :param axis: "x" or "y"
    """
    if axis=="x":
        xtick_max = ax.get_xlim()[1]
        xtick_min = ax.get_xlim()[0]
        step=(xtick_max-xtick_min)/width
        xbias = bias
        xticks = np.arange(xtick_min, xtick_max, step) + xbias
        return xticks
    elif axis=="y":
        ytick_max = ax.get_ylim()[1]
        ytick_min = ax.get_ylim()[0]   
        step=(ytick_max-ytick_min)/width
        ybias = bias
        yticks = np.arange(ytick_min, ytick_max, step) + ybias
        return yticks


def set_ticks_bar(ax:plt.Axes,axis:str="x",width:int=10,bias:float=0):
    """
    メモリの軸の数値の設定
    :param axis: "x" or "y"
    """
    if axis=="x":
        xticks = get_ticks_value(ax,width,bias,axis="x")
        ax.set_xticks(xticks)

    if axis=="y":
        yticks = get_ticks_value(ax,width,bias,axis="y")
        ax.set_yticks(yticks)


def ticks_value2label(ticks:list[float],config:dict):
    if config["ticks"]["num_type"] == "decimal":
        labels=[
            f"{(val):.{config['ticks']['num_decimal_places']}f}" for val in ticks
        ]
        return labels
    elif config["ticks"]["num_type"] == "scientific":
        labels = []
        for val in ticks:
            exponent = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
            base = val / 10**exponent
            labels.append(f"${base:.1f} \\times 10^{{{exponent}}}$")
        return labels


def set_ticks_label(ax:plt.Axes,config:dict):
    """
    メモリの軸の数値のラベルの設定
    """
    if not config["ticks"]["visible"]["xlabel"]:
        ax.set_xticklabels([])
    else:
        if config["limit"]["xmax"] is not None and config["limit"]["xmin"] is not None: 
            width=config["ticks"]["xwidth"]
            bias=config["ticks"]["xbias"]
            xticks = get_ticks_value(ax,width,bias,axis="x")
            ticks_labels=ticks_value2label(xticks,config)
            ax.set_xticklabels(ticks_labels,fontsize=config["ticks"]["fontsize"])  
        else:
            ax.tick_params(axis='x',labelsize=config["ticks"]["fontsize"])  


    if not config["ticks"]["visible"]["ylabel"]:
        ax.set_yticklabels([])
    else:
        if config["limit"]["ymax"] is not None and config["limit"]["ymin"] is not None: 
            width=config["ticks"]["ywidth"]
            bias=config["ticks"]["ybias"]
            yticks = get_ticks_value(ax,width,bias,axis="y")
            ticks_labels=ticks_value2label(yticks,config)
            ax.set_yticklabels(ticks_labels,fontsize=config["ticks"]["fontsize"])
        else:
            ax.tick_params(axis='y',labelsize=config["ticks"]["fontsize"])

   
def adjust_margins(fig, config):
    """
    Adjust the margins of the plot based on the configuration.
    """
    margins = config.get("canvas", {}).get("margin", {})
    left = cm2inch(margins.get("left", 0))
    right = cm2inch(margins.get("right", 0))
    top = cm2inch(margins.get("top", 0))
    bottom = cm2inch(margins.get("bottom", 0))

    # Adjust the subplot parameters
    fig.subplots_adjust(
        left=max(0, fig.subplotpars.left + left / fig.get_figwidth()),
        right=min(1, fig.subplotpars.right - right / fig.get_figwidth()),
        top=min(1, fig.subplotpars.top - top / fig.get_figheight()),
        bottom=max(0, fig.subplotpars.bottom + bottom / fig.get_figheight())
    )


def animate_volt_cmap(volt: np.ndarray, savepath: Path, filename="volt_animation", config: dict = {},fps=30):
    """
    :param volt: [neuron x timestep]
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    N, T = volt.shape

    # Extract configuration settings
    figsize = (
        cm2inch(config['canvas']['figsize']['width']), 
        cm2inch(config['canvas']['figsize']['height']),
    )
    font_family = config['canvas']['fontstyle']['family']
    title_fontsize = config['canvas']['title']['fontsize']
    label_fontsize = config['canvas']['label']['fontsize']
    x_label = config['canvas']['label']['xlabel']
    y_label = config['canvas']['label']['ylabel']
    x_min = config['canvas']['limit']['xmin'] if config['canvas']['limit']['xmin'] is not None else 0
    x_max = config['canvas']['limit']['xmax'] if config['canvas']['limit']['xmax'] is not None else T
    y_min = config['canvas']['limit']['ymin']
    y_max = config['canvas']['limit']['ymax']
    ticks_params=config["canvas"]["ticks"]

    # Heatmap settings
    cmap = config['heatmap']['cmap']
    is_transposed = config['heatmap']['is_transposed']
    colorbar_visible = config['heatmap']['colorbar']['visible']
    colorbar_label = config['heatmap']['colorbar']['label']['text']
    colorbar_label_fontsize = config['heatmap']['colorbar']['label']['fontsize']
    colorbar_ticks_fontsize = config['heatmap']['colorbar']['ticks']['fontsize']
    norm_type = config['heatmap']['colorbar']['ticks']['norm']
    vmin = config['heatmap']['colorbar']['ticks']['limit']['vmin']
    vmax = config['heatmap']['colorbar']['ticks']['limit']['vmax']

    plt.rcParams.update({'font.family': font_family})

    fig, ax = plt.subplots(figsize=figsize)

    norm = None
    if norm_type == 'symlog':
        norm = SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
    elif norm_type == 'log':
        norm = plt.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cax = ax.imshow(
        volt[:, 0].reshape(N, -1), cmap=cmap,
        interpolation="nearest", aspect='auto',
        norm=norm,
        extent=[0, T, -0.5, N-0.5]
    )

    if colorbar_visible:
        cbar = fig.colorbar(cax)
        cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize)
        cbar.ax.tick_params(labelsize=colorbar_ticks_fontsize)

    # ax.set_title("Voltage Colormap Animation", fontsize=title_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    set_ticks_bar(
        ax,"y",
        width=ticks_params["ywidth"],
        bias=ticks_params["ybias"]
    )
    set_ticks_label(ax,config["canvas"])
    # plt.tight_layout()
    adjust_margins(fig,config)

    plt.grid(alpha=config["canvas"]["grid"]["alpha"])

    def update(frame):
        frame_image=np.zeros_like(volt)
        frame_image[:,:frame]=volt[:,:frame]
        frame_image[:,frame:]=None
        cax.set_data(frame_image.reshape(N,-1))
        return cax,

    total_frames=int(T) 
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, blit=True, repeat=False
    )

    ani.save(savepath / f"{filename}.mp4", writer='ffmpeg',fps=fps,dpi=144)
    plt.close()


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--csvpath",type=str,required=True,help="csvのパスをそのままぶち込む")
    parser.add_argument("--fps",type=int,default=30,help="fps")
    args=parser.parse_args()

    csvpath=Path(args.csvpath)
    savepath=csvpath.parent
    filename=csvpath.stem 

    configpath=savepath / "config.yml" #configに設定をぶち込む
    with open(configpath, "r",encoding="utf-8") as f:
        config=yaml.safe_load(f)
    
    df=pd.read_csv(csvpath)
    animate_volt_cmap(df.values.T,savepath,filename,config=config,fps=args.fps)

    

if __name__=="__main__":
    main()