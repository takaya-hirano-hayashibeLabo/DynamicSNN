from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


from src.model import DynamicCSNN
# from src.model.dynamic_snn import DynamicSNN



def plot_results(s1,s2,v1, v2,v3 ,out_s2, out_s3, filename):

    plt.figure(figsize=(12, 12))
    T=s1.shape[0]
    T_scaled=v2.shape[0]

    plt.subplot(6,1,1)
    spike_times =torch.flatten(s1,start_dim=2).cpu().numpy()[:,0]
    print(s1.shape,spike_times.shape)
    plt.imshow(spike_times.T,interpolation="nearest", label=f'Base Spike Dim',cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,2)
    v1_ =torch.flatten(v1,start_dim=2).cpu().numpy()[:,0]  
    plt.imshow(v1_.T,interpolation="nearest",cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,3)
    v1_ =torch.flatten(v1,start_dim=2)
    v1_resampled=F.interpolate(v1_.permute(1,2,0), size=s2.shape[0], mode='linear', align_corners=False).permute(-1,0,1)
    v1_resampled=v1_resampled.cpu().numpy()[:,0]
    im3 = plt.imshow(v1_resampled.T,interpolation="nearest",cmap="viridis",aspect="auto")
    cbar3 = plt.colorbar(im3)  # Save the colorbar to use its limits
    plt.title('Scaled Base Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,4)
    spike_times =torch.flatten(s2,start_dim=2).cpu().numpy()[:,0]
    plt.imshow(spike_times.T,interpolation="nearest", cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Scaled In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,5)
    v2_ =torch.flatten(v2,start_dim=2).cpu().numpy()[:,0]
    im5 = plt.imshow(v2_.T,interpolation="nearest", cmap="viridis",aspect="auto", vmin=im3.get_clim()[0], vmax=im3.get_clim()[1])
    plt.colorbar(im5)  # Use the same colorbar limits
    plt.title('Dynamic CSNN Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))


    plt.subplot(6,1,6)
    v3_ =torch.flatten(v3,start_dim=2).cpu().numpy()[:,0]
    im6 = plt.imshow(v3_.T,interpolation="nearest", cmap="viridis",aspect="auto", vmin=im3.get_clim()[0], vmax=im3.get_clim()[1])
    plt.colorbar(im6)  # Use the same colorbar limits
    plt.title('Original Out Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.xlim(0,np.max([T_scaled,T]))
    

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")


    loss_dyna=np.mean((v1_resampled-v2_)**2)
    loss_org= np.mean((v1_resampled-v3_)**2)

    print(f"loss dyna : {loss_dyna.item()}, loss org : {loss_org.item()}, loss org/loss dyna: {(loss_org/loss_dyna).item()}")


def main():

    device = "cuda:0"
    with open('conf.yml', 'r') as file:
        config = yaml.safe_load(file)
    model=DynamicCSNN(conf=config["model"])
    model.to(device)
    model.eval()
    print(model)

    # Debugging parameters of each layer
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval()

    T = 100  # Number of timesteps
    batch_size = 16
    input_size = config["model"]["in-size"]
    in_channel=config["model"]["in-channel"]

    p=0.5
    base_input=torch.where(
        torch.rand(size=(T, batch_size, in_channel,input_size,input_size))<p,1.0,0.0
    ).to(device)
    # print(base_input)
    base_s,base_v=model(base_input)
    print(f"spike shape: {base_s.shape}")


    a=3.7  # 'a' can now be a float
    # Create scaled_input by shifting indices by a factor of 'a'
    scaled_input = torch.zeros(size=(int(a * T), batch_size, in_channel,input_size,input_size)).to(device)
    for t in range(T):
        scaled_index = int(a * t)
        if scaled_index < scaled_input.shape[0]:
            scaled_input[scaled_index] = base_input[t]

    org_s,org_v=model.forward(scaled_input)
    scaled_s,scaled_v=model.dynamic_forward_v1(scaled_input,a=torch.Tensor([a for _ in range(scaled_input.shape[0])]))

    plot_results(
        base_input.detach(),scaled_input.detach(),base_v.detach(),scaled_v.detach(),org_v.detach(),
        scaled_s,org_s,"membrane pattern")


if __name__=="__main__":
    main()