from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
import cv2

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import random_split
# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2
import time


# Datasets Preprocessing

# dataset_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_data/pose_label/label/test'
dataset_label_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_classify/data'

train_data = []
train_label = []
batch = 12
window_size = 12

for _, walk_item in enumerate(os.walk(dataset_label_path)):

    root, dirs, files = walk_item

    if not dirs and files:

        label_path = padarea_path = None

        for file in files:

            if "_area_label.csv" in file:
                label_path = os.path.join(root, file)
            elif "_area_data.xlsx" in file:
                padarea_path = os.path.join(root, file)


        if label_path and padarea_path:
            print("Now_padarea_path: ", padarea_path)
            ## 讀取動作區間標記
            label_df = pd.read_csv(f'{label_path}')
            label_df['label'] = label_df['label'].astype(int)

            padarea_df = pd.read_excel(f'{padarea_path}', sheet_name = 'Sheet1')


            batch = 12
            n = 1
            k = batch//2

            for _, row in label_df.iterrows():

                rowmid = (row['start'] + row['end'])//2
                pd_filter = padarea_df["Frame"].between(rowmid - n*k, rowmid + n*k)

                data_pos = padarea_df[pd_filter]
                window_usecols = ['Frame']
                # print(data_pos)

                for range_idx in range(n*2+1):

                    item_pos = np.array(data_pos.iloc[range_idx:range_idx+n*batch:n][window_usecols].values.flatten().tolist())

                    if len(item_pos) % batch != 0:
                        continue

                    print('range_idx:range_idx+n*batch:n: ', range_idx, range_idx+n*batch, n)
                    print('item_pos: ', item_pos)

                    item_pos = item_pos.reshape(batch, len(window_usecols), 1)
                    # print(item_pos)

                    if int(row['label']) in [1]:         # 推擋球
                        label = [0]
                    elif int(row['label']) in [2]:       # 切球
                        label = [1]

                    train_data.append(item_pos)
                    train_label.append(label)



                if row['end']+n*k >= len(padarea_df)/2:
                    continue





print("--------------------------------------------------------------------")
train_data = np.asarray(train_data).reshape(-1, batch, len(window_usecols) * 1)
np_train_data = train_data
print(np_train_data.shape)
np_train_label = np.asarray(train_label).reshape(-1, 1)
print(np_train_label.shape)
print(np_train_label)

# # Save the data as numpy arrays
np.save('/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_classify/training/ball_train_data.npy', np_train_data)
np.save('/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_classify/training/ball_train_label.npy', np_train_label)


