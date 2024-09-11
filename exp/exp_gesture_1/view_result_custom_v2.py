"""
v2:
init tauは考慮しないバージョン
"""

import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
EXP=Path(__file__).parent
import sys
sys.path.append(str(ROOT))

import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
from matplotlib import rcParams
import seaborn as sns
sns.set(style="darkgrid")


from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video,load_json2dict


def get_marker(data:pd.DataFrame,fig_conf:dict):

    marker=fig_conf["marker"]["middle"]

    sample=data["file"].values[0]
    if "large" in sample:
        marker=fig_conf["marker"]["large"]
    elif "small" in sample:
        marker=fig_conf["marker"]["small"]
    elif "lstm" in sample:
        marker=None

    return marker


def get_modelmarker(modeltype:str):

    if "dynamic-snn" in modeltype: return "o"
    elif "param" in modeltype: return "x"
    elif "snn" in modeltype: return "^"
    elif "lstm" in modeltype: return "s"


def plot_results(results_db, saveto: Path, fig_conf: dict):
    # フォント設定を変更
    rcParams['font.family'] = 'serif'
    # rcParams['font.serif'] = ['Times', 'Times New Roman', 'serif']


    unique_models = [label for label in results_db["model-type"].unique() if "small" in label or "lstm" in label] #tauinitがmiddleとlargeを除外
    
    fig, ax = plt.subplots(figsize=tuple(fig_conf["figsize"]))
    cmp = plt.get_cmap(fig_conf["color-map"])

    # モデルの種類ごとの色のlegend
    model_types = {model.split("_")[0] for model in unique_models}
    model_legend_handles = [
        plt.Line2D([0], [0], color=cmp(fig_conf["model-color"]["alpha"] * fig_conf["model-color"][model]), lw=4, label=fig_conf["model-label"][model])
        for model in model_types
    ]

    # tauのサイズごとのマーカーのlegend
    tau_sizes = ["large", "small", "middle"]
    tau_legend_handles = [
        plt.Line2D([0], [0], marker=fig_conf["marker"][size], color='black', linestyle='None', markersize=10, label=f'init {size} '+ r'$\tau$',markerfacecolor='none')
        for size in tau_sizes
    ]

    for model in unique_models:
        model_data = results_db[results_db["model-type"] == model]

        ax.plot(
            model_data["time-scale"], model_data["acc-mean"],
            marker=get_modelmarker(model),
            label=fig_conf["model-label"][model.split("_")[0]], color=cmp(fig_conf["model-color"]["alpha"] * fig_conf["model-color"][model.split("_")[0]]),
            markerfacecolor='none',
            markersize=fig_conf["plot"]["markersize"]
        )
        ax.fill_between(
            model_data["time-scale"],
            model_data["acc-mean"] - model_data["acc-std"],
            model_data["acc-mean"] + model_data["acc-std"],
            alpha=0.2,
            color=cmp(fig_conf["model-color"]["alpha"] * fig_conf["model-color"][model.split("_")[0]])
        )

    # ax.set_title('Accuracy vs Time-Scale')
    ax.set_xlabel('Time scale $a$'   ,fontsize=fig_conf["labelsize"])
    ax.set_ylabel('Accuracy',fontsize=fig_conf["labelsize"])
    ax.tick_params(axis='x', labelsize=fig_conf["legend"]["fontsize"])  # Set fontsize for x-ticks
    ax.tick_params(axis='y', labelsize=fig_conf["legend"]["fontsize"])  # Set fontsize for y-ticks
    ax.grid(True,alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(fontsize=fig_conf["legend"]["fontsize"])
    plt.tight_layout()

    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_result_custom_v2.svg"
    fig.savefig(savefile)  # Figureの外枠を削除
    plt.close(fig)
    

def plot_acc_delta(result_db_delta, saveto: Path,fig_conf:dict):
    unique_models = result_db_delta["model-type"].unique()
    time_scales = result_db_delta["time-scale"].unique()
    bar_width = 0.2
    gap = 0.1  # Gap between groups of bars
    index = np.arange(len(time_scales)) * (bar_width * len(unique_models) + gap)
    
    plt.figure(figsize=(14, 8))
    cmp=plt.get_cmap(fig_conf["color-map"])

    
    for i, model in enumerate(unique_models):
        model_data = result_db_delta[result_db_delta["model-type"] == model]
        plt.bar(
            index + i * bar_width, model_data["acc-mean-delta"], 
            bar_width, yerr=model_data["acc-std-delta"], capsize=5, label=model,
            color=cmp(fig_conf["model-color"]["alpha"]*fig_conf["model-color"][model.split("_")[0]])
            )
    
    plt.title('Accuracy Delta vs Time-Scale for All Models')
    plt.xlabel('Time Scale $a$')
    plt.ylabel('Accuracy Mean Delta')
    plt.xticks(index + bar_width * (len(unique_models) - 1) / 2, time_scales)
    plt.legend(title='Model Type')
    plt.grid(True)
    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_delta_result_costum.png"
    plt.savefig(savefile)
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",help="resultが入ったディレクトリ")
    args=parser.parse_args()


    fig_conf=load_yaml(Path(__file__).parent.parent/"model_view_conf.yml")
    print(fig_conf)


    results=[]
    dirs=os.listdir(Path(args.target))
    for file in dirs:
        filename=Path(args.target)/file/"result.json"

        try:
            js=load_json2dict(filename)
            # print(js)
            modeltype=js["model"]
            if "snn".casefold()==modeltype:
                modeltype="param-snn" if "param" in str(filename) else "snn"

            if "large".casefold() in str(filename):
                modeltype=modeltype+"_tau-large"
            elif "small".casefold() in str(filename):
                modeltype=modeltype+"_tau-small"

            results.append(
                [modeltype,js["time-scale"],js["acc_mean"],js["acc_std"],file]
            )

        except:
            pass

    results_db=pd.DataFrame(results,columns=["model-type","time-scale","acc-mean","acc-std","file"])
    results_db = results_db.sort_values(by=["model-type", "time-scale"])
    print(results_db)
    plot_results(saveto=Path(args.target),results_db=results_db,fig_conf=fig_conf)


    # Calculate the difference in acc-mean and acc-std for each model-type with time-scale=1 as the baseline
    baseline = results_db[results_db["time-scale"] == 1]
    result_db_delta = results_db.copy()
    result_db_delta["acc-mean-delta"] = result_db_delta.apply(
        lambda row: row["acc-mean"] - baseline[baseline["model-type"] == row["model-type"]]["acc-mean"].values[0],
        axis=1
    )
    result_db_delta["acc-std-delta"] = result_db_delta.apply(
        lambda row: np.sqrt(row["acc-std"]**2 + baseline[baseline["model-type"] == row["model-type"]]["acc-std"].values[0]**2),
        axis=1
    )
    print(result_db_delta)
    columns = ["dynamic-snn", "param-snn", "snn", "lstm"]
    filtered_result_db_delta = result_db_delta[result_db_delta["model-type"].isin(columns)]
    plot_acc_delta(filtered_result_db_delta, Path(args.target),fig_conf=fig_conf)


if __name__=="__main__":
    main()