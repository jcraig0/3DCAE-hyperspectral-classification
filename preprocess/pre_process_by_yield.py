import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import hyperspectral_datasets
import hyperspectral_datasets as HSI
from scipy.ndimage.interpolation import rotate


class HSI_preprocess:
    def __init__(self, name, dst_shape):
        self.name = name
        self.dst_shape = dst_shape

    def add_channel(self, data):
        if self.name is 'indian_pines':
            data_add_channel = np.zeros(self.dst_shape)
            print('After add channel to origin data, the data shape is: ',
                  data_add_channel.shape)
            data_add_channel[:, :, 0:103] = data[:, :, 0:103]
            data_add_channel[:, :, 109:149] = data[:, :, 104:144]
            data_add_channel[:, :, 164:219] = data[:, :, 145:200]
            return data_add_channel
        if self.name is 'salina':
            data_add_channel = np.zeros(self.dst_shape)
            print('After add channel to origin data, the data shape is: ',
                  data_add_channel.shape)
            data_add_channel[:, :, 0:107] = data[:, :, 0:107]
            data_add_channel[:, :, 112:153] = data[:, :, 107:148]
            data_add_channel[:, :, 167:223] = data[:, :, 148:204]
            return data_add_channel
        if self.name is 'paviau':
            data_add_channel = np.zeros(self.dst_shape)
            print('After add channel to origin data, the data shape is: ',
                  data_add_channel.shape)
            data_add_channel[:, :, :103] = data[:, :, :103]
            return data_add_channel
        if self.name is 'pavia':
            data_add_channel = np.zeros(self.dst_shape)
            print('After add channel to origin data, the data shape is: ',
                  data_add_channel.shape)
            data_add_channel[:, :, :102] = data[:, :, :102]
            return data_add_channel
        if self.name is 'acadia':
            data_add_channel = np.zeros(self.dst_shape)
            print('After add channel to origin data, the data shape is: ',
                  data_add_channel.shape)
            data_add_channel[:, :, :12] = data[:, :, :12]
            return data_add_channel

    # add zeros to make data easy to mean and var
    def data_add_zero(self, data, patch_size=5):
        assert data.ndim == 3
        dx = patch_size // 2
        data_add_zeros = np.zeros(
            (data.shape[0]+2*dx, data.shape[1]+2*dx, data.shape[2]))
        data_add_zeros[dx:-dx, dx:-dx, :] = data
        return data_add_zeros

    # get the mean or var of n*n
    def get_mean_data(self, data, patch_size=5, debug=False, var=False):
        assert isinstance(data.flatten()[0], float)
        dx = patch_size // 2
        # add zeros for mirror data
        data_add_zeros = self.data_add_zero(data=data, patch_size=patch_size)
        # get mirror date to calculate boundary pixel
        for i in range(dx):
            data_add_zeros[:, i, :] = data_add_zeros[:, 2 * dx - i, :]
            data_add_zeros[i, :, :] = data_add_zeros[2 * dx - i, :, :]
            data_add_zeros[:, -i - 1,
                           :] = data_add_zeros[:, -(2 * dx - i) - 1, :]
            data_add_zeros[-i - 1, :,
                           :] = data_add_zeros[-(2 * dx - i) - 1, :, :]
        if debug is True:
            print(data_add_zeros)
        data_mean = np.zeros(data.shape)
        data_var = np.zeros(data.shape)
        # get mean and var for evey patch
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                x_start, x_end = x, x+patch_size
                y_start, y_end = y, y+patch_size
                patch = np.array(
                    data_add_zeros[x_start:x_end, y_start:y_end, :])
                data_mean[x, y, :] = np.mean(patch.reshape(
                    patch_size**2, patch.shape[2]), axis=0)
                if var is True:
                    data_var[x, y, :] = np.std(patch.reshape(
                        patch_size**2, patch.shape[2]), axis=0)
        if var is False:
            return data_mean
        return np.concatenate((data_mean, data_var), axis=2)

    def get_patch_data(self, data, patch_size=5, debug=False, is_rotate=False):
        """
        :param data: m x n x c
        :param patch_size:
        :param debug:
        :param var:
        :return: m x n x patch_size x patch_size x c
        """
        assert isinstance(data.flatten()[0], float)
        dx = patch_size // 2
        # add zeros for mirror data
        data_add_zeros = self.data_add_zero(data=data, patch_size=patch_size)
        # get mirror date to calculate boundary pixel
        for i in range(dx):
            data_add_zeros[:, i, :] = data_add_zeros[:, 2 * dx - i, :]
            data_add_zeros[i, :, :] = data_add_zeros[2 * dx - i, :, :]
            data_add_zeros[:, -i - 1,
                           :] = data_add_zeros[:, -(2 * dx - i) - 1, :]
            data_add_zeros[-i - 1, :,
                           :] = data_add_zeros[-(2 * dx - i) - 1, :, :]
        if debug is True:
            print(data_add_zeros)
        # data_out = np.zeros(list(data.shape[:-1]) + [patch_size, patch_size] + [data.shape[-1],])
        # get mean and var for evey patch
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                x_start, x_end = x, x+patch_size
                y_start, y_end = y, y+patch_size
                patch = np.array(
                    data_add_zeros[x_start:x_end, y_start:y_end, :])
                if is_rotate:
                    # tmp: channels x patches x patches
                    tmp = patch.swapaxes(1, 2).swapaxes(0, 1)
                    rot_tmp = np.asarray(
                        [np.rot90(tmp, k=4, axes=(1, 2)) for i in range(4)])
                    # for i in range(4):
                    #     print(rot_tmp[i][0, :, :])
                    yield x, y, rot_tmp
                else:
                    yield x, y, patch.swapaxes(1, 2).swapaxes(0, 1)


def set_and_save_acadia_5d_data(patch_size=5, is_rotate=True):
    dataset = HSI.HSIDataSet('acadia')
    dataset.get_data()
    print('data shape is: ', dataset.data.shape)  # 4511, 975, 12

    data = np.array(dataset.data)
    dataset_process = HSI_preprocess(
        name='acadia', dst_shape=(4511, 975, 12))
    data = dataset_process.add_channel(data)
    data = dataset_process.data_add_zero(data)
    data_scale_to1 = data / np.max(data)
    data_5d = dataset_process.get_patch_data(
        data_scale_to1, patch_size=patch_size, is_rotate=is_rotate)

    [h, w, n_channels] = data_scale_to1.shape
    n_samples = h*w*4 if is_rotate else h*w
    if is_rotate:
        h5file_name = dataset.dir + \
            '/acadia_5d_patch_{}_with_rotate.h5'.format(patch_size)
    else:
        h5file_name = dataset.dir + '/acadia_5d_patch_{}.h5'.format(patch_size)

    file = h5py.File(h5file_name, 'w')
    file.create_dataset('data', shape=(n_samples, n_channels, patch_size, patch_size, 1),
                        chunks=(1024, n_channels, patch_size, patch_size, 1), dtype=np.float32,
                        maxshape=(None, n_channels, patch_size, patch_size, 1))
    file.close()

    with h5py.File(h5file_name, 'a') as h5f:
        for i, (x, y, patch) in enumerate(data_5d):
            if is_rotate:
                h5f['data'][4*i:4*(i+1)] = patch[:, :, :, :, None]
            else:
                h5f['data'][i] = patch[None, :, :, :, None]


if __name__ == '__main__':
    set_and_save_acadia_5d_data(patch_size=5, is_rotate=False)
    print("hello")
