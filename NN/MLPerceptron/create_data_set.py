import numpy as np
import pickle
from os import listdir, remove
from os.path import isfile, join
import urllib.request

# check if data exists, otherwise download
file_names = ['apple.npy', 'airplane.npy', 'basketball.npy', 'axe.npy', 'banana.npy', 'horse.npy', 'arm.npy', 'alarm clock.npy', 'ant.npy', 'bed.npy']
raw_data_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
raw_data_path = './'
for d in file_names:
    if not isfile(raw_data_path+d):
        urllib.request.urlretrieve(raw_data_url+d.replace(' ','%20'),raw_data_path+d)


data_size = 12500

data = np.load(raw_data_path + file_names[0])[:data_size]
label = np.zeros([np.load(raw_data_path + file_names[0]).shape[0], 1])[:data_size]
train_num_init = int(0.8*data.shape[0])

data_train = data[:train_num_init, :]
label_train = label[:train_num_init, :]

data_test = data[train_num_init:, :]
label_test = label[train_num_init:, :]

for idx, file_name in enumerate(file_names):
    if idx == 0:
        continue
    else:
        draw_data = np.load(raw_data_path + file_name)[:data_size]
        draw_label = idx * np.ones([draw_data.shape[0], 1])[:data_size]

        train_num = int(0.8*draw_data.shape[0])

        draw_data_for_train = draw_data[:train_num,:]
        draw_label_for_train = draw_label[:train_num, :]

        draw_data_for_test = draw_data[train_num:, :]
        draw_label_for_test = draw_label[train_num:, :]

        data_train = np.vstack([data_train, draw_data_for_train])
        label_train = np.vstack([label_train, draw_label_for_train])

        data_test = np.vstack([data_test, draw_data_for_test])
        label_test = np.vstack([label_test, draw_label_for_test])

with open('AI_quick_draw.pickle', 'wb') as save_ai_quick_draw:
    pickle.dump(np.uint8(data_train), save_ai_quick_draw)#, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_train, save_ai_quick_draw)#, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(np.uint8(data_test), save_ai_quick_draw)#, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_test, save_ai_quick_draw)#, protocol=pickle.HIGHEST_PROTOCOL)

# delete the raw data
for d in file_names:
    remove(raw_data_path+d)

