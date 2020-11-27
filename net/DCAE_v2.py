from __future__ import division, print_function, absolute_import

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, Conv3DTranspose, PReLU, BatchNormalization, MaxPool3D
from keras import backend as K
from keras import regularizers
import keras
import numpy as np
import h5py
from sklearn import preprocessing
import math
from keras.utils import plot_model


# indian_pines: 224 -> 201 -> 178 -> 9 -> 201 -> 224
# acadia: 12 -> 11 -> 10 -> 2 -> 11 -> 12
# prospect: 114 -> 102 -> 90 -> 6 -> 102 -> 114
def DCAE_v2(data_name, weight_decay=0.0005):
    model = Sequential()

    if data_name == 'indian_pines':
        dst_index = 0
    elif data_name == 'acadia':
        dst_index = 1
    else:
        dst_index = 2

    model.add(Conv3D(filters=24,
                     input_shape=((224, 12, 114)[dst_index], 5, 5, 1),
                     kernel_size=((24, 2, 13)[dst_index], 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv1"))
    model.add(BatchNormalization(name="BN1"))
    model.add(PReLU(name="PReLU1"))

    model.add(Conv3D(filters=48,
                     kernel_size=((24, 2, 13)[dst_index], 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv2"))
    model.add(BatchNormalization(name="BN2"))
    model.add(PReLU(name="PReLU2"))

    model.add(MaxPool3D(pool_size=((18, 5, 15)[dst_index], 1, 1),
                        strides=((18, 5, 15)[dst_index], 1, 1), name="Pool1"))

    model.add(Conv3DTranspose(filters=24,
                              kernel_size=((9, 10, 9)[dst_index], 3, 3),
                              kernel_regularizer=regularizers.l2(
                                  l=weight_decay),
                              strides=((22, 1, 17)[dst_index], 1, 1), name="Deconv1", padding='valid'))
    model.add(BatchNormalization(name="BN3"))
    model.add(PReLU(name="PReLU3"))
    model.add(Conv3DTranspose(filters=1,
                              kernel_size=((27, 2, 13)[dst_index], 3, 3),
                              kernel_regularizer=regularizers.l2(
                                  l=weight_decay),
                              strides=(1, 1, 1), name="Deconv2", padding='valid'))
    model.add(BatchNormalization(name="BN4"))
    return model


def DCAE_v2_feature(data_name, weight_decay=0.0005):
    model = Sequential()

    if data_name == 'indian_pines':
        dst_index = 0
    elif data_name == 'acadia':
        dst_index = 1
    else:
        dst_index = 2

    model.add(Conv3D(filters=24,
                     input_shape=((224, 12, 114)[dst_index], 5, 5, 1),
                     kernel_size=((24, 2, 13)[dst_index], 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv1"))
    model.add(BatchNormalization(name="BN1"))
    model.add(PReLU(name="PReLU1"))

    model.add(Conv3D(filters=48,
                     kernel_size=((24, 2, 13)[dst_index], 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv2"))
    model.add(BatchNormalization(name="BN2"))
    model.add(PReLU(name="PReLU2"))

    model.add(MaxPool3D(pool_size=((18, 5, 15)[dst_index], 1, 1),
                        strides=((18, 5, 15)[dst_index], 1, 1), name="Pool1"))
    return model
