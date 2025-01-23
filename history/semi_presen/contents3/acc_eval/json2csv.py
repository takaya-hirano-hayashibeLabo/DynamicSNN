
import os
import json
import pandas as pd
from pathlib import Path
PARENT=Path(__file__).parent

import argparse

def json_to_csv(directory, output_path,outname="results.csv"):
    data = []
    
    # JSONファイルを読み込む
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                json_data = json.load(file)
                data.append(json_data)


    #-- 生データ --------------------------------------------------
    # DataFrameに変換
    df = pd.DataFrame(data)
    # timescaleカラムで昇順に並び替え
    df = df.sort_values(by='time-scale')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # CSVファイルとして保存
    # df.to_csv(output_path/outname, index=False)


    #-- Δacc --------------------------------------------------
    df_reference=df[df["time-scale"]==1.0]
    delta_acc_mean=df["acc_mean"].values-df_reference["acc_mean"].values
    df["delta_acc_mean"]=delta_acc_mean

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # CSVファイルとして保存
    df.to_csv(output_path/outname, index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--jsonpath",default=PARENT/"results")
    args=parser.parse_args()

    json_to_csv(Path(args.jsonpath), Path(args.jsonpath),outname="results.csv")

if __name__=="__main__":
    main()