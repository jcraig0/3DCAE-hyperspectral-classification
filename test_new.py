import shapefile
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import sklearn

parser = argparse.ArgumentParser(description="test 3DCAE net",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default='prospect',
                    help='Name of dataset')
args = parser.parse_args()

if args.data == 'indian_pines':
    dst_shape = [145, 145]
elif args.data == 'acadia':
    dst_shape = [4511, 975]
else:
    dst_shape = [1534, 431]

features = h5py.File('./data/' + args.data + '_CAE_feature.h5', 'r')['feature']
features = np.squeeze(features)[:dst_shape[0] * dst_shape[1], :, :]
features = features.reshape(dst_shape + list(features.shape[-2:]))
print(features.shape)

sf = shapefile.Reader(
    shp=open('./shapefile/manual_ITC_extended.shp', 'rb'),
    dbf=open('./shapefile/manual_ITC_extended.dbf', 'rb'),
    shx=open('./shapefile/manual_ITC_extended.shx', 'rb')
)

shapes = sf.shapes()
records = sf.records()
base_coord = (144150, 920870)
res = 5
trees = ['red oak', 'snag', 'red spruce', 'black cherry', 'red maple', 'black birch', 'unk',
         'hemlock', 'white ash', 'white pine', 'yellow birch', 'norway spruce', 'American beech',
         'paper birch', 'blackgum', 'red pine']
map = np.zeros((round(1000 * res), round(1000 * res)))

for i in range(len(shapes)):
    new_shape = []
    for point in shapes[i].points:
        new_shape.append((round((point[0] - base_coord[0]) * res),
                          round((point[1] - base_coord[1]) * res)))
    new_shape = list(set(new_shape))
    for x, y in new_shape:
        map[x, y] = trees.index(records[i][0])

plt.imshow(map)
plt.show()