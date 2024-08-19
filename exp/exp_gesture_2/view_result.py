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


from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video,load_json2dict


def plot_results(results_db,saveto:Path):
    unique_models = results_db["model-type"].unique()
    plt.figure()
    for model in unique_models:
        model_data = results_db[results_db["model-type"] == model]
        plt.plot(model_data["delay"], model_data["acc-mean"], '-o', label=model)
        plt.fill_between(model_data["delay"], 
                         model_data["acc-mean"] - model_data["acc-std"], 
                         model_data["acc-mean"] + model_data["acc-std"], 
                         alpha=0.2)

    plt.title('Accuracy vs delay')
    plt.xlabel('delay')
    plt.ylabel('Accuracy Mean')
    plt.grid(True)
    plt.legend()

    savefile=saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto/"acc_result.png"
    plt.savefig(savefile)
    plt.close()


def plot_acc_delta(result_db_delta, saveto: Path):
    unique_models = result_db_delta["model-type"].unique()
    time_scales = result_db_delta["delay"].unique()
    bar_width = 0.2
    gap = 0.1  # Gap between groups of bars
    index = np.arange(len(time_scales)) * (bar_width * len(unique_models) + gap)
    
    plt.figure(figsize=(14, 8))
    
    for i, model in enumerate(unique_models):
        model_data = result_db_delta[result_db_delta["model-type"] == model]
        plt.bar(index + i * bar_width, model_data["acc-mean-delta"], bar_width, yerr=model_data["acc-std-delta"], capsize=5, label=model)
    
    plt.title('Accuracy Delta vs delay for All Models')
    plt.xlabel('delay')
    plt.ylabel('Accuracy Mean Delta')
    plt.xticks(index + bar_width * (len(unique_models) - 1) / 2, time_scales)
    plt.legend(title='Model Type')
    plt.grid(True)
    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_delta_result.png"
    plt.savefig(savefile)
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",help="resultが入ったディレクトリ")
    args=parser.parse_args()


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
                [modeltype,js["delay"],js["acc_mean"],js["acc_std"]]
            )

        except:
            pass

    results_db=pd.DataFrame(results,columns=["model-type","delay","acc-mean","acc-std"])
    results_db = results_db.sort_values(by=["model-type", "delay"])
    print(results_db)
    plot_results(saveto=Path(args.target),results_db=results_db)


    # Calculate the difference in acc-mean and acc-std for each model-type with delay=1 as the baseline
    baseline = results_db[results_db["delay"] == 0]
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
    plot_acc_delta(filtered_result_db_delta, Path(args.target))


if __name__=="__main__":
    main()