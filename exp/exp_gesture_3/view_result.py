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


def plot_results(results_db, saveto: Path):
    unique_models = results_db["model-type"].unique()
    unique_time_pairs = results_db["time-pair"].unique()
    bar_width = 0.2  # Width of each bar
    spacing = 3.0  # Further increased spacing between groups of bars
    index = [i * spacing for i in range(len(unique_time_pairs))]  # X locations for the groups

    plt.figure()
    
    for i, model in enumerate(unique_models):
        model_data = results_db[results_db["model-type"] == model]
        means = []
        stds = []
        for time_pair in unique_time_pairs:
            time_pair_data = model_data[model_data["time-pair"] == time_pair]
            if not time_pair_data.empty:
                means.append(time_pair_data["acc-mean"].values[0])
                stds.append(time_pair_data["acc-std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar([p + bar_width * i for p in index], means, bar_width, 
                yerr=stds, label=model, alpha=0.7)
    
    plt.title('Accuracy by Time-Pair', pad=30)  # Increase padding for the title
    plt.xlabel('Time-Pair')
    plt.ylabel('Accuracy Mean')
    plt.xticks([p + bar_width * (len(unique_models) / 2) for p in index], unique_time_pairs)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=3, borderaxespad=0.)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_result.png"
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()
            

def plot_acc_delta(result_db_delta, saveto: Path):
    unique_models = result_db_delta["model-type"].unique()
    time_pairs = result_db_delta["time-pair"].unique()  # Corrected column name
    bar_width = 0.2
    gap = 0.1  # Gap between groups of bars
    index = np.arange(len(time_pairs)) * (bar_width * len(unique_models) + gap)
    
    plt.figure(figsize=(14, 8))
    
    for i, model in enumerate(unique_models):
        model_data = result_db_delta[result_db_delta["model-type"] == model]
        plt.bar(index + i * bar_width, model_data["acc-mean-delta"], bar_width, yerr=model_data["acc-std-delta"], capsize=5, label=model)
    
    plt.title('Accuracy Delta vs Time-Pair for All Models')  # Updated title
    plt.xlabel('Time-Pair')  # Updated label
    plt.ylabel('Accuracy Mean Delta')
    plt.xticks(index + bar_width * (len(unique_models) - 1) / 2, time_pairs)
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

            time_pair=f"ts {js['timescale1']}-{js['timescale2']}"

            results.append(
                [modeltype,time_pair,js["acc_mean"],js["acc_std"]]
            )

        except:
            pass

    results_db=pd.DataFrame(results,columns=["model-type","time-pair","acc-mean","acc-std"])
    results_db_ = results_db[~results_db["time-pair"].str.contains("1-1")]  # Filter out "1-1"
    results_db_ = results_db_.sort_values(by=["model-type", "time-pair"])
    print(results_db_)
    plot_results(saveto=Path(args.target),results_db=results_db_)


    ## Calculate the difference in acc-mean and acc-std for each model-type with time-scale=1 as the baseline
    baseline = results_db[results_db["time-pair"] == "ts 1-1"]
    print("Baseline:\n", baseline)  # Debug print
    result_db_delta = results_db.copy()
    
    def calculate_delta(row):
        model_baseline = baseline[baseline["model-type"] == row["model-type"]]
        if not model_baseline.empty:
            acc_mean_delta = row["acc-mean"] - model_baseline["acc-mean"].values[0]
            acc_std_delta = np.sqrt(row["acc-std"]**2 + model_baseline["acc-std"].values[0]**2)
        else:
            acc_mean_delta = np.nan
            acc_std_delta = np.nan
        return pd.Series([acc_mean_delta, acc_std_delta])
    
    result_db_delta[["acc-mean-delta", "acc-std-delta"]] = result_db_delta.apply(calculate_delta, axis=1)
    
    print("Result DB Delta:\n", result_db_delta)  # Debug print
    columns = ["dynamic-snn", "param-snn", "snn", "lstm"]
    filtered_result_db_delta = result_db_delta
    # filtered_result_db_delta = result_db_delta[result_db_delta["model-type"].isin(columns)]
    print("Filtered Result DB Delta:\n", filtered_result_db_delta)  # Debug print
    plot_acc_delta(filtered_result_db_delta[~filtered_result_db_delta["time-pair"].str.contains("1-1")], Path(args.target))


if __name__=="__main__":
    main()