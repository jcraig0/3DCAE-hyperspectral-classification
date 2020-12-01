import shapefile
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import sklearn
import math

parser = argparse.ArgumentParser(description="test 3DCAE net",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default='prospect',
                    help='Name of dataset')
parser.add_argument('--swath', type=int,
                    help='Hyperspectral data swath number')
args = parser.parse_args()

if args.data == 'indian_pines':
    dst_shape = [145, 145]
elif args.data == 'acadia':
    dst_shape = [4511, 975]
else:
    dst_shape = [1500, [431, 436, 451, 483, 459, 487, 524, 566][int((args.swath - 5) / 2)]]

features = h5py.File('./data/' + args.data + \
    '{}_CAE_feature.h5'.format('_' + str(args.swath) if args.swath else ''), 'r')['feature']
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
base_coord = (144139, 920871)
size_x, size_y = 550, 750
scale = 1
trees = sorted(set([r[0] for r in records]))
map = np.zeros((round(size_y * scale), round(size_x * scale)))

for i in range(len(shapes)):
    new_shape = []
    for point in shapes[i].points:
        new_shape.append((round((point[0] - base_coord[0]) * scale),
                          round((point[1] - base_coord[1]) * scale)))
    new_shape = list(set(new_shape))
    color = trees.index(records[i][0]) + 1

    for j in range(len(new_shape)):
        x, y = new_shape[j][0], new_shape[j][1]
        map[x, y] = color

        # Draw line between vertices
        if j != len(new_shape) - 1:
            x_1, y_1 = new_shape[j+1][0], new_shape[j+1][1]
        else:
            x_1, y_1 = new_shape[0][0], new_shape[0][1]
        x_dist, y_dist = x_1 - x, y_1 - y
        x_dir, y_dir = 0, 0
        if abs(x_dist) >= abs(y_dist):
            distance = abs(x_dist)
            x_dir = 1 if x_dist > 0 else -1
        else:
            distance = abs(y_dist)
            y_dir = 1 if y_dist > 0 else -1
        for k in range(1, distance):
            if x_dir != 0:
                y_dir = (y_1 - y) / distance
            elif y_dir != 0:
                x_dir = (x_1 - x) / distance
            map[round(x + x_dir * k), round(y + y_dir * k)] = color

# Fill in holes in polygons
for y in range(round(size_y * scale)):
    for x in range(round(size_x * scale)):
        neighbors = []
        for dir in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                    (0, 1), (1, -1), (1, 0), (1, 1)):
            try:
                neighbors.append(map[y + dir[0], x + dir[1]])
            except IndexError:
                pass
        color = max(set(neighbors), key=neighbors.count)
        if color != 0:
            map[y, x] = color

plt.imshow(map)
plt.show()