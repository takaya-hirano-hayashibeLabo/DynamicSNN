from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import argparse
import numpy as np
import json
import os


from src.utils import load_json2dict


def main():

    resultpath=Path(__file__).parent/"result"
    resultdirs=[dir for dir in os.listdir(resultpath) if "trial" in dir]

    result_db=[]
    for resultdir in resultdirs:
        result_json=resultpath/resultdir/"result.json"
        result_dict=load_json2dict(result_json)
        result_db.append(result_dict)

    result_db=pd.DataFrame(result_db)
    print(result_db)

    # Set pandas display options for more precision
    pd.set_option('display.float_format', lambda x: '%.10f' % x)

    # Calculate and print mean and standard deviation for each column
    means = result_db.mean()
    stds = result_db.std()
    print("Means:\n", means)
    print("Standard Deviations:\n", stds)


if __name__=="__main__":
    main()