import torch
import numpy as np

def encode2spike(f, thresholds):
    """
    連続値の時系列データをスパイク列に変換する関数
    :param f: 連続値の時系列データ (torch tensor) [N x T x m]
    :param thresholds: スパイクを生成する閾値のリスト
    :return: スパイク列 (torch tensor) [N x T x c x m]
    """
    N, T, m = f.shape
    c = len(thresholds)
    
    # スパイク列を初期化
    o = torch.zeros((N, T, c, m), device=f.device)
    
    # 閾値をテンソルに変換
    thresholds_tensor = torch.tensor(thresholds, device=f.device).view(1, 1, c, 1)
    
    # fをシフトして前後の値を比較
    f_prev = f[:, :-1, :].unsqueeze(2)  # [N, T-1, 1, m]
    f_next = f[:, 1:, :].unsqueeze(2)   # [N, T-1, 1, m]
    
    # 閾値を超えた場所を検出
    spike_mask = ((f_prev <= thresholds_tensor) & (f_next > thresholds_tensor)) | ((f_prev >= thresholds_tensor) & (f_next < thresholds_tensor))
    
    # スパイク列を更新
    o[:, 1:, :, :] = spike_mask.float()
    
    return o


if __name__ == "__main__":
    """
    テスト項目：
    ・numpyによるループ処理とtorchによるバッチ処理の一致(テスト通過)
    ・スパイク数がタイムスケール変化によらない
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    t = np.linspace(0, 10, 100)
    f = np.sin(t)
    f = torch.tensor(f)
    f=f.unsqueeze(0).unsqueeze(-1) #[n x t x m]
    th = np.linspace(-1, 1, 100)
    o = encode2spike(f, th) # [n x t x c x m]
    print(f"f shape: {f.shape}, o shape: {o.shape}")

    a=0.5
    t_scaled=t/a
    f_scaled=np.sin(t_scaled)
    f_scaled=torch.tensor(f_scaled)
    f_scaled=f_scaled.unsqueeze(0).unsqueeze(-1) #[n x t x m]
    o_scaled=encode2spike(f_scaled, th)

    # 描画
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(t_scaled, f.squeeze().numpy())
    axs[2].plot(t_scaled, f_scaled.squeeze().numpy())
    # スパイク列を描画
    o = o.squeeze().numpy()
    for k, threshold in enumerate(th):
        spike_times = np.where(o[:, k] == 1)[0]
        axs[1].scatter(t_scaled[spike_times], [k] * len(spike_times), label=f'Threshold {threshold}')
    o_scaled = o_scaled.squeeze().numpy()
    for k, threshold in enumerate(th):
        spike_times = np.where(o_scaled[:, k] == 1)[0]
        axs[3].scatter(t_scaled[spike_times], [k] * len(spike_times), label=f'Threshold {threshold}')
    plt.savefig(Path(__file__).parent / "spike_sequence.png")
    plt.close()


    #>> スパイクの総数が時間変化によらない >>
    base_sequence=60
    print(f"original out spike num:{np.sum(o[:base_sequence])}")
    print(f"scaled out spike num:{np.sum(o_scaled[:int(base_sequence*a)])}")
    #>> スパイクの総数が時間変化によらない >>


