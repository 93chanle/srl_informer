import argparse
import os
import torch
import yaml

import numpy as np
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

from exp.exp_informer import Exp_Informer
from data.data_loader import Dataset_Custom

data_set = Dataset_Custom(
            root_path='./data/processed/SRL/',
            data_path='SRL_NEG_00_04.csv',
            flag='val',
            size=[96, 48, 3],
            features='S',
            target='capacity_price',
            inverse=False,
            timeenc=1,
            freq='d',
            cols=None
        )