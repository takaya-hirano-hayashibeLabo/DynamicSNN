from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import os
from snntorch import functional as SF
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression


from src.utils import load_yaml,load_hdf5,print_terminal,CustomDataset,scale_sequence,resample_scale


# ヒートマップを描画して保存する関数
def save_heatmap(result, savepath):
    result_pd = pd.DataFrame(result, columns=["scale", "window", "firing rate"])
    pivot_table = result_pd.pivot(index="window", columns="scale", values="firing rate")  # 縦軸をwindowの値に設定
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_table, aspect='auto', cmap='hot', origin='lower', extent=[result_pd["scale"].min(), result_pd["scale"].max(), result_pd["window"].min(), result_pd["window"].max()])
    plt.colorbar(label='Firing Rate')
    plt.xlabel('Scale')
    plt.ylabel('Window')
    plt.title('Firing Rate Heatmap')
    plt.savefig(savepath / "firing_rate_heatmap.png")
    plt.close()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str,help="train-dataの名前",required=True)
    parser.add_argument("--dt",type=float,required=True)
    args = parser.parse_args()


    savepath=Path(__file__).parent/"result"/args.dataname
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    datapath=str(ROOT/f"train-data/{args.dataname}")
    test_files=[f"{datapath}/test/{file}" for file in os.listdir(f"{datapath}/test")]
    in_test,target_test=load_hdf5(test_files,num_workers=32) #[batch x time-sequence x ...], [batch]


    result=[]
    a_min,a_max,a_step=1,50,1
    w_min,w_max,w_step=10,150,10
    for a in np.arange(a_min,a_max,a_step):
        print_terminal(contents=f"[scale a={a:.2f}] "+"-"*300)

        a_list=a*np.ones(len(in_test[0])+1)
        in_test_a=scale_sequence(np.array(in_test),a=a_list,dt=args.dt) #[batch x time-sequence]

        for w in range(w_min,w_max,w_step):
            in_test_window=in_test_a[:,:w]
            fr=np.mean(in_test_window,axis=1) #時間方向に平均
            fr=np.mean(fr,axis=tuple([i+1 for i in range(fr.ndim-1)])) #空間方向に平均
            fr=np.mean(fr) #バッチ方向に平均

            result.append([a,w,fr])

        result_pd=pd.DataFrame(result,columns=["scale","window","firing rate"])
        result_pd.to_csv(savepath/"fr_scale_heat.csv",index=False)

    save_heatmap(result,savepath)


    # 線形回帰を行う
    regression_results = []
    plt.figure(figsize=(12, 8))  # グラフのサイズを設定

    for window in result_pd["window"].unique():
        subset = result_pd[result_pd["window"] == window]
        X = np.log(subset["firing rate"].values.reshape(-1, 1) + 1e-10)  # 入力に対数変換
        y = np.log(subset["scale"].values + 1e-10)  # 出力に対数変換

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_
        
        # 予測値と2乗誤差を計算
        predictions = model.predict(X)
        mse = np.mean((predictions - y) ** 2)  # 2乗誤差
        regression_results.append([window, slope, intercept, mse])

    # 2乗誤差でソートし、上位5つを選択
    regression_df = pd.DataFrame(regression_results, columns=["window", "slope", "intercept", "mse"])
    top_windows = regression_df.nsmallest(5, 'mse')

    # グラフ描画
    for window in top_windows["window"]:
        subset = result_pd[result_pd["window"] == window]
        X = np.log(subset["firing rate"].values.reshape(-1, 1) + 1e-10)
        y = np.log(subset["scale"].values + 1e-10)

        model = LinearRegression()
        model.fit(X, y)

        # 回帰曲線を描画
        plt.scatter(subset["firing rate"].values, subset["scale"].values, label=f'Window {window}', alpha=0.5)
        plt.plot(np.exp(X), np.exp(model.predict(X)), color='red')  # 回帰直線を描画

    plt.xlabel('Firing Rate')
    plt.ylabel('Scale')
    plt.title('Transformed Linear Regression for Top 5 Windows by MSE')
    plt.legend()
    plt.grid()
    plt.savefig(savepath / "log_linear_regression_curves.png")  # グラフを保存
    plt.close()
    regression_df.to_csv(savepath/"linear_regression_results.csv", index=False)  # 結果をCSVに保存


if __name__=="__main__":
    main()