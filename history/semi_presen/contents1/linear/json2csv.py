# dev/workspace/prj_202409_2MR/laplace-version/history/semi_presen/contents1/json2csv.py

import os
import json
import pandas as pd
from pathlib import Path
PARENT=Path(__file__).parent

def json_to_csv(directory, output_file):
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
    # CSVファイルとして保存
    df.to_csv(output_file, index=False)

# 使用例
json_to_csv(PARENT/'results', PARENT/'results.csv')