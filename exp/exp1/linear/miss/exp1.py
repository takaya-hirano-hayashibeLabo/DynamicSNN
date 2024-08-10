from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F


from src.dynamic_snn import DynamicSNN


def plot_results(base_v, scaled_v, filename):
    plt.figure(figsize=(12, 6))

    color=["orange","blue"]
    # Plot base_v
    plt.subplot(1, 2, 1)
    for dim in range(base_v.shape[2]):
        plt.plot(base_v[:, 0, dim].cpu().numpy(), label=f'Base Voltage Dim {dim}',color=color[dim])
    plt.title('Base Input Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()

    # Plot scaled_v
    plt.subplot(1, 2, 2)
    for dim in range(scaled_v.shape[2]):
        plt.plot(scaled_v[:, 0, dim].cpu().numpy(), label=f'Scaled Voltage Dim {dim}',color=color[dim])
    plt.title('Scaled Input Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")


def plot_results2(s1,s2,v1, v2,v3, filename):

    plt.figure(figsize=(12, 8))
    color=["orange","blue","green"]

    plt.subplot(5,1,1)
    s1_resampled=F.interpolate(s1.permute(1,2,0), size=s2.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    for dim in range(s1.shape[2]):
        plt.plot(s1[:, 0, dim].cpu().numpy(), label=f'Base Spike Dim {dim}',color=color[0])
        plt.plot(s1_resampled[:, 0, dim].cpu().numpy(), label=f'Base Spike Dim {dim} (resampled)',color=color[0], ls="--")
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

    plt.subplot(5,1,3)
    # s2_resampled=F.interpolate(s2.permute(1,2,0), size=s1.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    for dim in range(s1.shape[2]):
        plt.plot(s2[:, 0, dim].cpu().numpy(), label=f'Scaled Spike Dim {dim}',color=color[1])
        # plt.plot(s2_resampled[:, 0, dim].cpu().numpy(), label=f'Scaleld Spike Dim {dim} (resampled)',color=color[1],ls="--")
    plt.title('Scaled Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.legend()
    plt.xlim(0,v2.shape[0])

    plt.subplot(5,1,4)
    # v2_resampled=F.interpolate(v2.permute(1,2,0), size=s1.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    for dim in range(v1.shape[2]):
        plt.plot(v2[:, 0, dim].cpu().numpy(), label=f'Scaled Voltage Dim {dim}',color=color[1])
        # plt.plot(v2_resampled[:, 0, dim].cpu().numpy(), label=f'Scaled Voltage Dim {dim} (resampled)',color=color[1],ls="--")
    plt.title('Scaled Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.xlim(0,v2.shape[0])


    plt.subplot(5,1,5)
    # v3_resampled=F.interpolate(v3.permute(1,2,0), size=s1.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    for dim in range(v1.shape[2]):
        plt.plot(v3[:, 0, dim].cpu().numpy(), label=f'Original Voltage Dim {dim}',color=color[2])
        # plt.plot(v3_resampled[:, 0, dim].cpu().numpy(), label=f'Original Voltage Dim {dim} (resampled)',color=color[2],ls="--")
    plt.title('Original Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.xlim(0,v2.shape[0])
    

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")



def main():

    with open('conf.yml', 'r') as file:
        config = yaml.safe_load(file)
    model=DynamicSNN(conf=config["model"])
    model.eval()

    T=50
    batch=1
    insize=config["model"]["in-size"]

    base_input=torch.ones(size=(T,batch,insize))
    base_input[int(T/3):]=0.0
    base_s,base_v=model(base_input)



    a=2
    scaled_input=torch.ones(size=(int(a*T),batch,insize))

    scaled_input[int(a*T/3):]=0.0
    org_s,org_v=model.forward(scaled_input)
    scaled_s,scaled_v=model.dynamic_forward(scaled_input,a=torch.Tensor([a for _ in range(scaled_input.shape[0])]))

    # # Resample to match the length of base_input
    # import torch.nn.functional as F
    # org_v=org_v.permute(1,2,0)
    # scaled_v=scaled_v.permute(1,2,0)
    # resampled_org_v = F.interpolate(org_v, size=T, mode='linear', align_corners=False).permute(-1,0,1)
    # resampled_scaled_v = F.interpolate(scaled_v, size=T, mode='linear', align_corners=False).permute(-1,0,1)
    # org_v=org_v.permute(-1,0,1)
    # scaled_v=scaled_v.permute(-1,0,1)

    # plot_results(base_v.detach(),resampled_scaled_v.detach(),filename="base_vs_scaled")
    # plot_results(base_v.detach(),resampled_org_v.detach(),filename="base_vs_org")
    # plot_results(base_v.detach(),scaled_v.detach(),filename="base_vs_scaled")
    # plot_results(base_v.detach(),org_v.detach(),filename="base_vs_org")

    plot_results2(base_input.detach(),scaled_input.detach(),base_v.detach(),scaled_v.detach(),org_v.detach(),"aaa")

if __name__=="__main__":
    main()