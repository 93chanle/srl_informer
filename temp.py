import argparse
import os
import torch
import yaml

import numpy as np
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

from exp.exp_informer import Exp_Informer
from data.data_loader import Dataset_Informer, Dataset_XGB

root_path = os.path.normpath("data/dummy_dataset")

data_set = Dataset_XGB(
            root_path=root_path,
            data_path='SRL_NEG_00_04_dummy.csv',
            flag='test',
            size=[4,1],
            features='S',
            target='capacity_price',
            inverse=False,
            timeenc=1,
            freq='d',
            scale=None,
            cols=['gas', 'coal']
        )

print('finish')