"""
Author: Armin Berger
Date: 10.01.23

Practice Project which seeks to use Pytorch and FluidML to classify SMS text
messages as spam (1) or not (0).
"""

import pandas as pd
import os

current_dir = os.getcwd()





pd.read_csv(f"{current_dir}/data/train.csv")