import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
import seaborn as sns
sns.set(style="darkgrid")
import numpy as np

class ColorMap:
    def __init__(self):
        self.cmap={
            "spike":(255,192,0),
            "v_base":(21,96,130),
            "v_lif":(127,127,127),
            "v_dyna":(120,32,110)
            }
        
    def __call__(self,key):
        return np.array(self.cmap[key])/255


def plot_results(s1,s2,v1, v2,v3, filename):
    plt.rcParams['font.family'] = 'serif'
    cmap=ColorMap()

    figsize=(16, 10)
    fontsize=20
    labelsize=16
    plt.figure(figsize=figsize)

    plt.subplot(5,1,1)
    for dim in range(s1.shape[2]):
        spike_times = torch.nonzero(s1[:, 0, dim], as_tuple=True)[0].cpu().numpy()
        plt.vlines(spike_times, ymin=0, ymax=1, color=cmap("spike"), label=f'Base Spike Dim {dim}')
        plt.title('Input spikes $o(t)$', fontsize=fontsize)
    # plt.xlabel('Time')
    plt.ylabel('Spike', fontsize=fontsize)
    plt.xlim(0,v2.shape[0])
    plt.tick_params(axis='x', labelbottom=False) 
    plt.yticks(fontsize=labelsize)
    # plt.legend()

    plt.subplot(5,1,2)
    v1_resampled=F.interpolate(v1.permute(1,2,0), size=s2.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    for dim in range(v1.shape[2]):
        plt.plot(v1[:, 0, dim].cpu().numpy(), label=f'Base Voltage Dim {dim}',color=cmap("v_base"))
        # plt.plot(v1_resampled[:, 0, dim].cpu().numpy(), label=f'Base Voltage Dim {dim}',color=color[0],ls="--")
    plt.title('Membrane potential $v(t)$', fontsize=fontsize)
    # plt.xlabel('Time')
    plt.ylabel('Voltage', fontsize=fontsize)
    # plt.legend()
    plt.tick_params(axis='x', labelbottom=False) 
    plt.yticks(fontsize=labelsize)
    plt.xlim(0,v2.shape[0])
    base_voltage_ylim = plt.ylim()  # Get y-axis limits for v1

    plt.subplot(5,1,3)
    plt.plot(v1_resampled[:, 0, 0].cpu().numpy(), label=f'Base Voltage Dim {dim}',color=cmap("v_base"),ls="--")
    plt.title("Ideal membrane potential $v(at)$", fontsize=fontsize)
    plt.ylabel("Voltage", fontsize=fontsize)
    plt.tick_params(axis='x', labelbottom=False) 
    plt.yticks(fontsize=labelsize)
    plt.xlim(0,v2.shape[0])

    plt.subplot(5,1,4)
    for dim in range(s1.shape[2]):
        spike_times = torch.nonzero(s2[:, 0, dim], as_tuple=True)[0].cpu().numpy()
        plt.vlines(spike_times, ymin=0, ymax=1, color=cmap("spike"), label=f'Scaled Spike Dim {dim}')
    plt.title('Scaled spikes $o(at)$', fontsize=fontsize)
    # plt.xlabel('Time')
    plt.ylabel('Spike', fontsize=fontsize)
    plt.tick_params(axis='x', labelbottom=False) 
    plt.yticks(fontsize=labelsize)
    # plt.legend()
    plt.xlim(0,v2.shape[0])

    plt.subplot(5,1,5)
    for dim in range(v1.shape[2]):
        plt.plot(v3[:, 0, dim].cpu().numpy(), label='$v_{LIF}(t)$',color=cmap("v_lif"))
    # plt.title('LIF membrane potential $v_{LIF}(t)$', fontsize=fontsize)
    # # plt.xlabel('Time')
    # plt.ylabel('Voltage', fontsize=fontsize)
    # # plt.legend()
    # plt.xlim(0,v2.shape[0])
    # plt.ylim(base_voltage_ylim)  # Set y-axis limits to match v1
    # plt.xticks([])
    # plt.yticks(fontsize=labelsize)
    
    for dim in range(v1.shape[2]):
        plt.plot(v2[:, 0, dim].cpu().numpy(), label='$v_{proposed}(t)$',color=cmap("v_dyna"))
    plt.title('Membrane potential', fontsize=fontsize)
    plt.xlabel('timestep $t$', fontsize=fontsize)
    plt.ylabel('Voltage', fontsize=fontsize)
    # plt.legend()
    plt.xlim(0,v2.shape[0])
    plt.ylim(base_voltage_ylim)  # Set y-axis limits to match v1
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend(loc='upper right',fontsize=labelsize)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}")
    plt.close()