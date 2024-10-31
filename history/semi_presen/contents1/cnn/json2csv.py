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
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                json_data = json.load(file)
                data.append(json_data)

    # DataFrameに変換
    df = pd.DataFrame(data)
    # timescaleカラムで昇順に並び替え
    df = df.sort_values(by='timescale')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # CSVファイルとして保存
    df.to_csv(output_path/outname, index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default=PARENT/"results")
    parser.add_argument("--output",default=PARENT)
    args=parser.parse_args()

    json_to_csv(Path(args.input), Path(args.output))

if __name__=="__main__":
    main()