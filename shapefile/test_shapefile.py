import shapefile
import matplotlib.pyplot as plt
import numpy as np

sf = shapefile.Reader(
    shp=open('manual_ITC_extended.shp', 'rb'),
    dbf=open('manual_ITC_extended.dbf', 'rb'),
    shx=open('manual_ITC_extended.shx', 'rb')
)

shapes = sf.shapes()
records = sf.records()
base_coord = (144150, 920870)
trees = ['red oak', 'snag', 'red spruce', 'black cherry', 'red maple', 'black birch', 'unk',
         'hemlock', 'white ash', 'white pine', 'yellow birch', 'norway spruce', 'American beech',
         'paper birch', 'blackgum', 'red pine']
map = np.zeros((1000, 1000))

for i in range(len(shapes)):
    new_shape = []
    for point in shapes[i].points:
        new_shape.append((round(point[0] - base_coord[0]), round(point[1] - base_coord[1])))
    new_shape = list(set(new_shape))
    for x, y in new_shape:
        map[x, y] = trees.index(records[i][0])

plt.imshow(map)
plt.show()