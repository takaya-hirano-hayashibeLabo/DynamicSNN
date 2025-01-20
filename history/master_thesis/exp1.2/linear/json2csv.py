# dev/workspace/prj_202409_2MR/laplace-version/history/semi_presen/contents1/json2csv.py

import os
import json
import pandas as pd
from pathlib import Path
PARENT=Path(__file__).parent

import argparse

def json_to_csv(directory, output_path,outname="results.csv"):
    data = []
    
    # JSONファイルを読み込む
    for dir in os.listdir(directory):
        try:
            result_data=json.load(open(directory/dir/"result.json"))
            args=json.load(open(directory/dir/"args.json"))

            data.append([
                args["timescale"], args["tau"],
                result_data["lif_mean"], result_data["lif_std"],
                result_data["dyna_mean"], result_data["dyna_std"],
            ])
        except:
            print(f"no json file in {directory/dir}")

    # DataFrameに変換
    df = pd.DataFrame(data,columns=["timescale","tau","lif_mean","lif_std","dyna_mean","dyna_std"])
    # timescaleカラムで昇順に並び替え
    df = df.sort_values(by='timescale')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # CSVファイルとして保存
    df.to_csv(output_path/outname, index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default=PARENT/"results")
    args=parser.parse_args()

    json_to_csv(Path(args.input), Path(args.input))

if __name__=="__main__":
    main()