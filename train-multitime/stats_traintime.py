import pandas as pd
from datetime import datetime
import argparse

def main(csv_path):
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)

    # datetime列をdatetime型に変換
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 学習時間の計算
    df['epoch_time'] = df['datetime'].diff().fillna(pd.Timedelta(seconds=0))

    # 分単位に変換
    df['epoch_time_minutes'] = df['epoch_time'].dt.total_seconds() / 60

    # 最初のエポックの時間を除外
    epoch_times = df['epoch_time_minutes'][1:]

    # 平均と標準偏差の計算
    mean_epoch_time = epoch_times.mean()*60
    std_epoch_time = epoch_times.std()*60

    # 結果の表示
    print(f"1エポックあたりの学習時間の平均: {mean_epoch_time} s")
    print(f"1エポックあたりの学習時間の標準偏差: {std_epoch_time} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average and standard deviation of epoch times from a CSV file.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    main(args.csv_path)