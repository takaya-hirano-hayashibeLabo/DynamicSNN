from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))
import os

import torch
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
from src.model import DynamicCSNN,CSNN,DynamicResCSNN



def plot_weight_histograms_recursive(model, layer_index=0, base_name="layer", saveto="."):
    # Create the directory if it doesn't exist
    os.makedirs(saveto, exist_ok=True)

    # Iterate over the layers in the model
    for name, layer in model.named_children():
        print(name, layer)
        # If the layer is another module, recurse into it
        if isinstance(layer, torch.nn.Module):
            # Check if the layer has weights
            if hasattr(layer, 'weight') and layer.weight is not None:
                print(layer.weight)  # Print the weights
                weights = layer.weight.data.cpu().numpy().flatten()
                
                # Plot the histogram
                plt.figure(figsize=(6, 4))
                plt.hist(weights, bins=30, alpha=0.7, color='blue')
                plt.title(f'Layer {layer_index+1} - {layer.__class__.__name__} Weight Distribution')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
                plt.grid(True)
                
                # Save the plot as a .png file
                layer_name = f"{base_name}_{layer_index+1}_{layer.__class__.__name__}"
                file_name = os.path.join(saveto, f"{layer_name}_weight_distribution.png")
                plt.savefig(file_name)
                plt.close()
            
            # Recurse into the layer
            plot_weight_histograms_recursive(layer, layer_index, base_name, saveto)
        
        # Increment the layer index
        layer_index += 1



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",default="dyna-snn")
    parser.add_argument("--device",default=0,type=int)
    args=parser.parse_args()

    device = f"cuda:{args.device}"
    with open(Path(args.target)/'conf.yml', 'r') as file:
        config = yaml.safe_load(file)

    
    model=DynamicResCSNN(conf=config["model"])
    # model.load_state_dict(torch.load(Path(args.target)/f"result/model_best.pth",map_location=device))
    model.to(device)
    model.eval()
    # print(model)


    result_path=Path(__file__).parent/"result"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    plot_weight_histograms_recursive(model,saveto=result_path)


if __name__=="__main__":
    main()