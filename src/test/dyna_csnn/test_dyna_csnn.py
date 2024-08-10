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
import sys
MODELDIR=Path(__file__).parent.parent
sys.path.append(str(MODELDIR))
import torch
from model.dynamic_snn import DynamicCSNN,DynamicLIF


def main():

    # Load configuration
    with open(Path(__file__).parent/"conf.yml", "r") as file:
        conf = yaml.safe_load(file)

    # Create a random input tensor
    T = 300  # Number of timesteps
    batch_size = 128
    input_size = conf["model"]["in-size"]
    in_channel=conf["model"]["in-channel"]
    device = "cuda:0"

    input_data = torch.randn(T, batch_size, in_channel,input_size,input_size).to(device)

    # Initialize the model
    model = DynamicCSNN(conf["model"])
    model.to(device)
    print(model)


    print("test 'forward' func"+"-"*100)
    # Forward pass
    out_s, out_v = model(input_data)

    # Print the shapes of the outputs
    print("Output spikes shape:", out_s.shape)
    print("Output voltages shape:", out_v.shape)

    # Print the size of tau for each DynamicLIF layer
    for layer in model.model:
        if isinstance(layer, DynamicLIF):
            print("Tau size:", layer.w.size())

    # Save the model
    model_path = Path(__file__).parent / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Load the model
    loaded_model = DynamicCSNN(conf["model"])
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    print("Model loaded successfully")

    # Verify the loaded model
    out_s_loaded, out_v_loaded = loaded_model(input_data)
    print("Output spikes shape (loaded model):", out_s_loaded.shape)
    print("Output voltages shape (loaded model):", out_v_loaded.shape)

    # Print the size of tau for each DynamicLIF layer in the loaded model
    for layer in loaded_model.model:
        if isinstance(layer, DynamicLIF):
            print("Tau size (loaded model):", layer.w.size())



    print("\ntest 'dynamic_forward' func"+"-"*100)
    # Dynamic forward pass
    a = torch.rand(T).to(device)  # Random speed multipliers
    out_s_dyn, out_v_dyn = model.dynamic_forward_v1(input_data, a)

    # Print the shapes of the dynamic outputs
    print("Dynamic output spikes shape:", out_s_dyn.shape)
    print("Dynamic output voltages shape:", out_v_dyn.shape)

    # Print the size of tau for each DynamicLIF layer after dynamic forward
    for layer in model.model:
        if isinstance(layer, DynamicLIF):
            print("Tau size (dynamic forward):", layer.w.size())


if __name__=="__main__":
    main()