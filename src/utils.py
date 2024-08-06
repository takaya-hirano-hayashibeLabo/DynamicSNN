import h5py
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def print_terminal(pre="\033[96m",contents="",suffi="\033[0m"):
    """
    :param pre:接頭
    :param contentns: 末尾をカットされてもいいcontents
    :param suffi: 接尾
    """

    import shutil
    termianl_width=shutil.get_terminal_size().columns
    s= contents[:termianl_width] if len(contents)>termianl_width else contents

    print(pre+s+suffi)
    

def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



def load_single_hdf5(path):
    with h5py.File(path, 'r') as f:
        data = f['events'][:]
        target = f['target'][()]
    return data, target

def load_hdf5(file_path_list: list, num_workers: int = 64):
    """
    pathのリストからhdf5ファイルを読み込み, データを返す
    :return datas: [batch x time_sequence x ...]
    :return targets: [batch]
    """
    with Pool(num_workers) as pool:
        results = pool.map(load_single_hdf5, file_path_list)

    datas, targets = zip(*results)
    return list(datas), list(targets)



def resample_scale(a: list, target_length: int) -> list:
    """
    1次元のリストを指定した長さにリサンプリングする関数
    線形補間を使用
    :param data: リサンプリングする1次元のリスト
    :param target_length: リサンプリング後の長さ
    :return: リサンプリングされたリスト
    """
    original_length = len(a)
    if original_length == target_length:
        return a

    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)
    
    resampled_data = np.interp(target_indices, original_indices, a)
    
    return resampled_data.tolist()

def spike2timestamp(spike:np.ndarray,dt:float):
    """
    0か1のスパイク列をスパイクのタイムスタンプに変換する関数
    :param spike: [time-sequence x ...]
    :return timestamp: スパイクのある時刻 
    :return idx_x: スパイクある空間位置
    """
    idx_sp=np.argwhere(spike==1)
    idx_t=idx_sp[:,0]
    idx_x=idx_sp[:,1:]
    timestamp=dt*idx_t
    return timestamp.reshape(-1,1),idx_x


def timestamp2spike(timestamp:np.ndarray,idx_x,dt,spike_shape:tuple):
    """
    :param spike_shape: 時間以外の次元のサイズ
    """
    from math import ceil

    T=ceil(np.max(timestamp)/dt)+1
    idx_time=np.array((timestamp/dt).round(),dtype=np.int64)

    spike=np.zeros(shape=(T,*spike_shape))
    spike_idx=np.concatenate([idx_time,idx_x],axis=1)
    spike[tuple(spike_idx.T)]=1

    return spike


def scale_sequence(data:np.ndarray,a:list,dt:float):
    """
    :param data: [batch x time-sequence x ...]
    :param a: スケーリングのリスト. 1以上[time-sequence]
    :param dt: dataの⊿t
    """
    from math import ceil

    elapsed = np.cumsum(np.concatenate(([0], a[:-1]))) * dt
    T_max=ceil(elapsed[-1]/dt)
    scaled_data=[]
    for data_i in tqdm(data): #バッチ一つずつ処理する
        timestamp, idx_x=spike2timestamp(data_i,dt)

        scaled_timestamp = np.zeros_like(timestamp)

        for t in range(data_i.shape[0]):
            mask = (timestamp >= t * dt) & (timestamp < (t + 1) * dt)
            scaled_timestamp[mask] = elapsed[t]

        scaled_spike=timestamp2spike(
            scaled_timestamp,idx_x,
            dt,data_i.shape[1:]
        )

        if scaled_spike.shape[0]<T_max:
            scaled_spike=np.concatenate([
                scaled_spike, np.zeros(shape=(T_max-scaled_spike.shape[0], *data_i.shape[1:]))
                ],axis=0)

        scaled_data.append(scaled_spike)

    return np.array(scaled_data)

