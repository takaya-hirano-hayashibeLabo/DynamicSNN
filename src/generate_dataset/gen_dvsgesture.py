import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import os
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm
import tonic
from math import ceil
import time
import concurrent.futures


def save_to_hdf5(file_path, data, target):
    """
    hdf5で圧縮して保存
    そうしないとTオーダのデータサイズになってしまう
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('events', data=data, compression='gzip')
        f.create_dataset('target', data=target)


def process_window(events, t_window_start, t_window_end, resolution):
    events_i = events[(t_window_start <= events["t"]) & (events["t"] < t_window_end)]
    new_events_i = []

    if len(events_i) == 0:
        return new_events_i

    for polarity in [True, False]:
        events_polarity = events_i[events_i["p"] == polarity]
        if len(events_polarity) > 0:
            x_mean = int(np.mean(events_polarity["x"]))
            y_mean = int(np.mean(events_polarity["y"]))
            new_events_i.append((x_mean, y_mean, polarity, int(t_window_start + resolution / 2)))

    return new_events_i

def process_window_batch(events, t_window_starts, t_window_ends, resolution):
    new_events_batch = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_window, events, t_window_start, t_window_end, resolution)
            for t_window_start, t_window_end in zip(t_window_starts, t_window_ends)
        ]

        for future in concurrent.futures.as_completed(futures):
            new_events_batch.extend(future.result())

    return new_events_batch

def change_resolution(events: np.ndarray, resolution, num_workers=64):
    """
    処理の分割はマルチプロセス、処理はマルチスレッドで行うことで高速に処理が行える
    詳しい調査はこちら : https://www.notion.so/DVSGesture-27bb047931454bc5a5fdb142b36f150f?pvs=4#a480ffdb10cc4ce6a0ea3baba8e30611
    :param events: [event_num x (x,y,t,p)]
    """
    t_start = events[0]["t"]
    t_end = events[-1]["t"]
    event_length = ceil((t_end - t_start) / resolution)

    new_events = []

    batch_size = int(event_length/np.max([num_workers,16]))  # バッチサイズはprocess数がnum_workerと同じくらいになるように設定する
    t_window_starts = [t_start + i * resolution for i in range(event_length)]
    t_window_ends = [t_start + (i + 1) * resolution for i in range(event_length)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_window_batch, events, t_window_starts[i:i + batch_size], t_window_ends[i:i + batch_size], resolution)
            for i in range(0, event_length, batch_size)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            new_events.extend(future.result())

    dtype = np.dtype([('x', '<i8'), ('y', '<i8'), ('p', '<i8'), ('t', '<i8')])
    new_events = np.array(new_events, dtype=dtype)

    # tを基準にソート
    new_events.sort(order='t')

    # print("original events shape: ", events.shape)
    # print(events[:20])
    # print("new events shape: ", new_events.shape)
    # print(new_events[:20])

    return new_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="dvsgesture-v1")
    parser.add_argument("--resolution", default=20, type=float, help="分解能を何倍に落とすか. 20なら20μsに落とす")
    parser.add_argument("--num_worker",default=8,type=int, help="使うコア数")
    args = parser.parse_args()

    trainset = tonic.datasets.DVSGesture(save_to=str(ROOT / "original-data"), train=True)
    testset = tonic.datasets.DVSGesture(save_to=str(ROOT / "original-data"), train=False)

    start_time = time.time()


    event_savepath=ROOT/f"custum-data/DVSGesture/{args.filename}"
    if not os.path.exists(event_savepath):
        os.makedirs(event_savepath/"train")
        os.makedirs(event_savepath/"test")


    i_max=4
    print("generate train data...")
    for i in range(len(trainset)):
        print(f"[{i+1}/{len(trainset)}]")
        events = trainset[i][0]
        new_events = change_resolution(events, args.resolution,args.num_worker)

        save_to_hdf5(
            event_savepath/f"train/data{i}.h5",new_events,trainset[i][1]
        )

        if i==i_max:
            break
    print("\033[92mdone\033[0m\n")


    print("generate test data...")
    for i in range(len(testset)):
        print(f"[{i+1}/{len(testset)}]")
        events = testset[i][0]
        new_events = change_resolution(events, args.resolution,args.num_worker)

        save_to_hdf5(
            event_savepath/f"test/data{i}.h5",new_events,testset[i][1]
        )

        if i==i_max:
            break
    print("\033[92mdone\033[0m")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken for processing the loop: {elapsed_time:.2f} seconds")


if __name__=="__main__":
    main()