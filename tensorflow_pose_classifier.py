from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob
import tensorflow
import os
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models
from tensorflow.keras.models import Model

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
# from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from tensorflow.keras.utils import to_categorical


# Verify Images
# image_list = []
# for filename in glob.glob('./table-tennis/label_mask/merge_R/*.png'):
#     im = Image.open(filename)
#     image_list.append(im)

# # print(image_list)

# image_list_0 = np.array(image_list[0])
# print(image_list_0.shape)


# Datasets Preprocessing
dataset_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_data/pose_label/label/test'

train_data = []
train_label = []
batch=12
window_size = 12
for _, walk_item in enumerate(os.walk(dataset_path)):

    root, dirs, files = walk_item

    if not dirs and files:

        label_path = None

        for file in files:

            if "_label.csv" in file:
                label_path = os.path.join(root, file)

        if label_path:
            ## 讀取動作區間標記
            label_df = pd.read_csv(f'{label_path}')
            label_df['label'] = label_df['label'].astype(int)

            n = 4
            k = batch//2
            ## 動作區間篩選
            for _, row in label_df.iterrows():
                ## 在動作的Frame區間 且 屬於動作者的骨幹資料
                # if row['label'] in range(1, 5):
                #     pd_filter_1 = label_df['Person_id'] == 0
                # elif row['label'] in range(5, 9):
                #     pd_filter_1 = label_df['Person_id'] == 1

                # if row['label'] in [1, 5]:
                #     label = [1]
                # elif row['label'] in [2, 6]:
                #     label = [2]
                # elif row['label'] in [3, 4, 7, 8]:
                #     label = [0]


                # if row['label'] in range(1, 5):
                #     pd_filter_1 = label_df['Person_id'] == 0
                # elif row['label'] in range(5, 9):
                #     pd_filter_1 = label_df['Person_id'] == 1

                if int(row['label']) in [5]:         # 右正手發球
                    label = [1]
                elif int(row['label']) in [6]:       # 右反手發球
                    label = [2]
                elif int(row['label']) in [7]:       # 右正手回球
                    label = [3]
                elif int(row['label']) in [8]:       # 右反手回球
                    label = [4]
                else:
                    label = [0]


                for i in range((row['end'] - row['start'] - window_size) + 1):
                    print('group_num', (row['end'] - row['start'] - window_size) + 1)
                    img_list = []
                    begin = row['start'] + i
                    for j in range (begin, begin + 12):
                        # img_list.append(cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R/merge_R_frame_{j}.png'))
                        img = cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R/merge_R_frame_{j}.png')
                        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                        img_list.append(img)


                    print("start: ", begin)
                    print("end: ", begin + 12)
                    print("label: ", label)
                    print(row['label'])
                    print(len(img_list))
                    img_list_np = np.asarray(img_list)
                    print(img_list_np.shape)

                    train_data.append(img_list)
                    train_label.append(label)

            # train_data.append("/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R/", )
            # train_label.append(label)


print("--------------------------------------------------------------------")
np_train_data = np.asarray(train_data).reshape(-1, window_size, 256, 256, 3)
print(np_train_data.shape)
np_train_label = np.asarray(train_label).reshape(-1, 1)
print(np_train_label.shape)
print(np_train_label)


# train_data = np.asarray(train_data).reshape(-1, batch, (1080, 1920))          # resize image shape
# train_label = np.asarray(train_label).reshape(-1, 1)







# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
#             tf.keras.layers.Conv1D(16, 3, strides=1, padding='same', activation='relu', data_format='channels_first'),
#             tf.keras.layers.LayerNormalization(),
#             tf.keras.layers.Conv1D(32, 3, strides=1, padding='same', activation='relu', data_format='channels_first'),
#             tf.keras.layers.Conv1D(64, 3, strides=1, padding='same', activation='relu', data_format='channels_first'),
#             tf.keras.layers.Conv1D(64, 4, strides=2, padding='same', activation='relu', data_format='channels_first'),
#             tf.keras.layers.LayerNormalization(),
#             tf.keras.layers.Conv1D(128, 4, strides=2, padding='same', activation='relu', data_format='channels_first'),
#             tf.keras.layers.Conv1D(128, 4, strides=2, padding='same', activation='relu', data_format='channels_first'),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(num_classes, activation='softmax'),
#         ]
#     )
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#     )






# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(100, activation='relu'))
# model.add(layers.Dense(11, activation='softmax'))
