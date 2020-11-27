# -*-coding:utf-8-*-
from __future__ import division, print_function, absolute_import

import numpy as np
import h5py
import time
import os
import scipy.io as sio
import tensorflow as tf
from net.DCAE_v2 import DCAE_v2_feature as DCAE_fea
from net.DCAE_v2 import DCAE_v2 as DCAE
from evaluate.PSNR import psnr
from sklearn.model_selection import StratifiedKFold
# import preprocess.preprocess as pre
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse
import keras
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n_class = 17


def mkdir_if_not_exist(the_dir):
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)


def get_predict(model, x_data, shape, batch_size=32, ):
    shape = list(shape)
    nums = x_data.shape[0]
    shape[0] = nums
    x_predict = np.zeros(shape)
    for i in np.arange(int(nums / batch_size)):
        x_start, x_end = i*batch_size, (i+1)*batch_size
        data_temp = x_data[x_start:x_end, :, :, :, :]
        x_predict[x_start:x_end, :, :, :, :] = model.predict(data_temp)
        # print(x_start, x_end)
    if nums % batch_size:
        x_start, x_end = nums - nums % batch_size, nums
        data_temp = x_data[x_start:x_end, :, :, :, :]
        x_predict[x_start:x_end, :, :, :, :] = model.predict(data_temp)
    return x_predict


def pre_process_data(data_name):
    h5file_name = os.path.expanduser(
        './hyperspectral_datas/{}/data/{}_5d_patch_5.h5'.format(data_name))
    file = h5py.File(h5file_name, 'r')
    data = np.array(file['data'])
    return data


def pre_process_indian_for_cls():
    """
    """
    h5file_name = os.path.expanduser(
        './hyperspectral_datas/indian_pines/data/indian_5d_patch_5.h5')
    file = h5py.File(h5file_name, 'r')
    X = np.array(file['data'])
    y = np.array(file['labels']).flatten()
    X = X[y != 0]
    y = y[y != 0]
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)
    split_gen = skf.split(X, y)
    (train_index, test_index) = next(split_gen)
    X_train, X_test = X[test_index], X[train_index]
    y_train, y_test = y[test_index], y[train_index]
    y_train = to_categorical(y_train, n_class)
    y_test = to_categorical(y_test, n_class)
    return X_train, y_train, train_index, X_test, y_test, test_index


def train_3DCAE_v3(x_train, x_test, data_name, model_name):
    model = DCAE(data_name, weight_decay=0.0005)
    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['MSE'])
    n_epoch = args.epoch
    save_model_per_epoch = 50
    save_times = n_epoch / save_model_per_epoch
    for i in range(int(save_times)):
        model.fit(x_train, x_train, epochs=save_model_per_epoch, shuffle=True,
                  validation_data=(x_test, x_test),
                  verbose=1,
                  batch_size=32)
        x_predict = get_predict(
            model=model, x_data=x_train, shape=x_train.shape)
        x_true = np.asarray(x_train)
        x_pred_centre = x_predict[:, :, 2, 2, :]
        x_true_centre = x_true[:, :, 2, 2, :]
        psnr(x_true_centre, x_pred_centre)
        mkdir_if_not_exist(os.path.split(model_name)[0])
        model.save('{}_{}.model'.format(
            model_name, save_model_per_epoch*(i+1)))


def train_v3(data_name, model_name):
    x_train = pre_process_data(data_name)
    X_train, X_test = train_test_split(x_train, train_size=0.9)
    train_3DCAE_v3(X_train, X_test, data_name, model_name=model_name)


class ModeTest:
    def __init__(self, model_name, save_file_name, epoch=40):
        self.data, self.label = self.pre_process_data()
        self.model_name = model_name+'_{}.model'.format(epoch)
        self.epoch = epoch
        self.feature = None
        self.save_file_name = save_file_name

    def pre_process_data(self):
        h5file_name = os.path.expanduser(
            './hyperspectral_datas/indian_pines/data/indian_5d_patch_5.h5')
        file = h5py.File(h5file_name, 'r')
        data = np.array(file['data'])
        labels = np.array(file['labels']).flatten()
        return data, labels

    def get_feature(self):
        feature_model = DCAE_fea()
        feature_model.load_weights(self.model_name, by_name=True)
        self.feature = get_predict(
            model=feature_model, x_data=self.data, shape=feature_model.predict(self.data[:2]).shape)
        print(self.feature.shape)
        plt.imshow(self.feature[:, 1, 0, 0, 1].reshape((145, 145)))
        plt.show()

    def get_global_feature(self, global_data):
        feature_model = DCAE_fea()
        feature_model.load_weights(self.model_name, by_name=True)
        self.feature = feature_model.predict(global_data)
        plt.imshow(self.feature[:, 1, 0, 0, 1].reshape((145, 145)))
        plt.show()

    def save_feature_label(self):
        file = h5py.File(self.save_file_name, 'w')
        file.create_dataset('feature', data=self.feature)
        file.create_dataset('label', data=self.label)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train 3DCAE net",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='prospect',
                        help='Name of data folder')
    parser.add_argument('--mode', type=str, default='train',
                        help='train, test.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='1000 is ok')
    args = parser.parse_args()
    if args.mode == 'train':
        model_name = './model/trained_by_' + args.data + '/CAE/DCAE_v2_epoch_'
        train_v3(data_name, model_name=model_name)
    elif args.mode == 'test':
        model_name = './model/trained_by_' + args.data + '/CAE/DCAE_v2_epoch_'
        test_mode = ModeTest(model_name=model_name,
                             save_file_name='./data/' + args.data + '_CAE_feature.h5',
                             epoch=args.epoch)
        test_mode.get_feature()
        test_mode.save_feature_label()
