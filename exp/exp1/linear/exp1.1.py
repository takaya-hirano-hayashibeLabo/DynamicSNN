from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F


from src.model import DynamicSNN
# from src.model.dynamic_snn import DynamicSNN



def plot_results2(s1,s2,v1, v2,v3, filename):

    plt.figure(figsize=(12, 8))
    color=["orange","blue","green"]

    plt.subplot(5,1,1)
    for dim in range(s1.shape[2]):
        spike_times = torch.nonzero(s1[:, 0, dim], as_tuple=True)[0].cpu().numpy()
        plt.vlines(spike_times, ymin=0, ymax=1, color=color[0], label=f'Base Spike Dim {dim}')
        plt.title('In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,v2.shape[0])
    plt.legend()


    plt.subplot(5,1,2)
    v1_resampled=F.interpolate(v1.permute(1,2,0), size=s2.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    for dim in range(v1.shape[2]):
        plt.plot(v1[:, 0, dim].cpu().numpy(), label=f'Base Voltage Dim {dim}',color=color[0])
        plt.plot(v1_resampled[:, 0, dim].cpu().numpy(), label=f'Base Voltage Dim {dim}',color=color[0],ls="--")
    plt.title('Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.xlim(0,v2.shape[0])
    base_voltage_ylim = plt.ylim()  # Get y-axis limits for v1

    plt.subplot(5,1,3)
    for dim in range(s1.shape[2]):
        spike_times = torch.nonzero(s2[:, 0, dim], as_tuple=True)[0].cpu().numpy()
        plt.vlines(spike_times, ymin=0, ymax=1, color=color[1], label=f'Scaled Spike Dim {dim}')
    plt.title('Scaled Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.legend()
    plt.xlim(0,v2.shape[0])

    plt.subplot(5,1,4)
    for dim in range(v1.shape[2]):
        plt.plot(v2[:, 0, dim].cpu().numpy(), label=f'Scaled Voltage Dim {dim}',color=color[1])
    plt.title('Scaled Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.xlim(0,v2.shape[0])
    plt.ylim(base_voltage_ylim)  # Set y-axis limits to match v1


    plt.subplot(5,1,5)
    for dim in range(v1.shape[2]):
        plt.plot(v3[:, 0, dim].cpu().numpy(), label=f'Original Voltage Dim {dim}',color=color[2])
    plt.title('Original Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.xlim(0,v2.shape[0])
    plt.ylim(base_voltage_ylim)  # Set y-axis limits to match v1
    

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")



def main():

    with open('conf.yml', 'r') as file:
        config = yaml.safe_load(file)
    model=DynamicSNN(conf=config["model"])

    # Debugging parameters of each layer
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval()

    T=250
    batch=1
    insize=config["model"]["in-size"]

    p=0.1
    base_input=torch.where(
        torch.randn(size=(T,batch,insize))<p,1.0,0.0
    )
    # print(base_input)
    base_s,base_v=model(base_input)


    a=3  # 'a' can now be a float
    # Create scaled_input by shifting indices by a factor of 'a'
    scaled_input = torch.zeros(size=(int(a * T), batch, insize))
    for t in range(T):
        scaled_index = int(a * t)
        if scaled_index < scaled_input.shape[0]:
            scaled_input[scaled_index] = base_input[t]

    org_s,org_v=model.forward(scaled_input)
    scaled_s,scaled_v=model.dynamic_forward_v1(scaled_input,a=torch.Tensor([a for _ in range(scaled_input.shape[0])]))

    plot_results2(base_input.detach(),scaled_input.detach(),base_v.detach(),scaled_v.detach(),org_v.detach(),"aaa")

if __name__=="__main__":
    main()