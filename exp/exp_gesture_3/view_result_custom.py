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
import seaborn as sns
sns.set(style="darkgrid")
from matplotlib import rcParams
rcParams['font.family'] = 'serif'



from src.utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video,load_json2dict


def get_hatch(data:pd.DataFrame,fig_conf:dict):

    hatch=fig_conf["hatch"]["middle"]

    sample=data["file"].values[0]
    if "large" in sample:
        hatch=fig_conf["hatch"]["large"]
    elif "small" in sample:
        hatch=fig_conf["hatch"]["small"]
    elif "lstm" in sample:
        hatch=None

    return hatch


def get_color(model, fig_conf):
    cmp = plt.get_cmap(fig_conf["color-map"])
    base_model = model.split("_")[0]
    return cmp(fig_conf["model-color"]["alpha"] * fig_conf["model-color"][base_model])



def plot_results(results_db, saveto: Path,fig_conf:dict):
    unique_models = results_db["model-type"].unique()
    unique_time_pairs = results_db["time-pair"].unique()
    bar_width = 0.2  # Width of each bar
    spacing = 3.0  # Further increased spacing between groups of bars
    index = [i * spacing for i in range(len(unique_time_pairs))]  # X locations for the groups

    plt.figure(figsize=tuple(fig_conf["figsize"]))
    cmp = plt.get_cmap(fig_conf["color-map"])

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
        
        plt.bar(
            [p + bar_width * i for p in index], means, bar_width, 
            yerr=stds, label=model, alpha=0.7,
            color=cmp(fig_conf["model-color"]["alpha"] * fig_conf["model-color"][model.split("_")[0]]),
            hatch=get_hatch(model_data,fig_conf)
            )
    
    # plt.title('Accuracy by Time-Pair', pad=30)  # Increase padding for the title
    plt.xlabel('Time-Pair')
    plt.ylabel('Accuracy Mean')
    plt.xticks([p + bar_width * (len(unique_models) / 2) for p in index], unique_time_pairs)
    plt.grid(True,alpha=0.5)
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=3, borderaxespad=0.)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.7])  # プロットのレイアウトを調整


    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_result_custom.png"
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

    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_delta_result_custom.png"
    plt.savefig(savefile)
    plt.close()


def plot_results_facetgrid(results_db, saveto: Path, fig_conf: dict):
    # Create a new column for tau size based on the model type
    def get_tau_size(model_type):
        if "large" in model_type:
            return "large"
        elif "small" in model_type:
            return "small"
        else:
            return "middle"

    results_db["tau-size"] = results_db["model-type"].apply(get_tau_size)
    results_db["base-model"] = results_db["model-type"].apply(lambda x: x.split("_")[0])

    # Set up the FacetGrid with specified column order
    col_order = ["dynamic-snn", "param-snn", "snn", "lstm"]
    g = sns.FacetGrid(results_db, row="time-pair", col="base-model", col_order=col_order, height=fig_conf["figsize"][1] / 4, aspect=1, sharey=True)
    
    # Define the bar plot function
    def barplot(data, **kwargs):
        base_model = data["base-model"].unique()[0]
        bar_width = 0.175
        spacing = 0.05  # Spacing between bars
        if base_model in ["snn", "param-snn", "dynamic-snn"]:
            for i, tau_size in enumerate(["small", "middle", "large"]):
                tau_data = data[data["tau-size"] == tau_size]
                if not tau_data.empty:
                    hatch = fig_conf["hatch"][tau_size]
                    color = get_color(base_model, fig_conf)
                    x_positions = np.arange(len(tau_data)) + (i - 1) * (bar_width + spacing)  # Center the bars
                    plt.bar(x_positions, tau_data["acc-mean"], yerr=tau_data["acc-std"], capsize=5, hatch=hatch, color=color, width=bar_width, alpha=0.9)
        else:
            hatch = get_hatch(data, fig_conf)
            color = get_color(base_model, fig_conf)
            x_positions = np.arange(len(data))
            plt.bar(x_positions, data["acc-mean"], yerr=data["acc-std"], capsize=5, hatch=hatch, color=color, width=bar_width, alpha=0.9)

    # Map the bar plot to the FacetGrid
    g.map_dataframe(barplot)

    # Customize the plot
    g.set_titles("")
    g.set_xticklabels(rotation=45)
    g.fig.tight_layout(rect=[0, 0, 1, 1])

    # Add grid lines
    for ax in g.axes.flat:
        ax.grid(True, alpha=0.4, zorder=-1)

    # Add column and row labels with background color
    for ax, col in zip(g.axes[0], g.col_names):
        col_label=fig_conf["model-label"][col]
        ax.annotate(col_label, xy=(0.5, 1.1), xycoords='axes fraction', ha='center', va='center', size='medium')

    for ax, row in zip(g.axes[:, -1], g.row_names):
        row_label = "timescale" + "\n    " + (row.split(" "))[-1]
        ax.annotate(row_label, xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center', size='medium', rotation=-90)

    # Add y-axis label to the leftmost plot
    g.set_axis_labels("", "Accuracy")
    g.axes[0, 0].set_ylabel("Accuracy")

    # Add legend for tau sizes
    tau_sizes = ["large", "small", "middle"]
    tau_legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', hatch=fig_conf["hatch"][size]*4, label=f'init {size}'+r' $\tau$')
        for size in tau_sizes
    ]
    tau_legend = g.fig.legend(handles=tau_legend_handles, bbox_to_anchor=(0.5, 1.075), loc='upper center', frameon=False, ncol=3, fontsize=fig_conf["legend"]["fontsize"])
    g.fig.add_artist(tau_legend)

    # Save the plot
    savefile = saveto if ".png" in str(saveto) or ".jpg" in str(saveto) else saveto / "acc_result_custom_facetgrid.png"
    g.savefig(savefile)
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

            time_pair=f"ts {js['timescale1']}-{js['timescale2']}"

            results.append(
                [modeltype,time_pair,js["acc_mean"],js["acc_std"],file]
            )

        except:
            pass

    results_db=pd.DataFrame(results,columns=["model-type","time-pair","acc-mean","acc-std","file"])
    results_db_ = results_db[~results_db["time-pair"].str.contains("1-1")]  # Filter out "1-1"
    results_db_ = results_db_.sort_values(by=["model-type", "time-pair"])
    print(results_db_)
    plot_results(saveto=Path(args.target),results_db=results_db_,fig_conf=fig_conf)


    plot_results_facetgrid(results_db_, Path(args.target), fig_conf)


    # ## Calculate the difference in acc-mean and acc-std for each model-type with time-scale=1 as the baseline
    # baseline = results_db[results_db["time-pair"] == "ts 1-1"]
    # print("Baseline:\n", baseline)  # Debug print
    # result_db_delta = results_db.copy()
    
    # def calculate_delta(row):
    #     model_baseline = baseline[baseline["model-type"] == row["model-type"]]
    #     if not model_baseline.empty:
    #         acc_mean_delta = row["acc-mean"] - model_baseline["acc-mean"].values[0]
    #         acc_std_delta = np.sqrt(row["acc-std"]**2 + model_baseline["acc-std"].values[0]**2)
    #     else:
    #         acc_mean_delta = np.nan
    #         acc_std_delta = np.nan
    #     return pd.Series([acc_mean_delta, acc_std_delta])
    
    # result_db_delta[["acc-mean-delta", "acc-std-delta"]] = result_db_delta.apply(calculate_delta, axis=1)
    
    # print("Result DB Delta:\n", result_db_delta)  # Debug print
    # columns = ["dynamic-snn", "param-snn", "snn", "lstm"]
    # filtered_result_db_delta = result_db_delta
    # # filtered_result_db_delta = result_db_delta[result_db_delta["model-type"].isin(columns)]
    # print("Filtered Result DB Delta:\n", filtered_result_db_delta)  # Debug print
    # plot_acc_delta(filtered_result_db_delta[~filtered_result_db_delta["time-pair"].str.contains("1-1")], Path(args.target))


if __name__=="__main__":
    main()