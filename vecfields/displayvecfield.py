from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
import sys

def trimshape(arr, newshape):
    oldshape = arr.shape
    idxs = []
    for i in range(len(oldshape)):
        even = np.linspace(0, oldshape[i]-1, newshape[i], dtype=np.int32)
        paddedshape = tuple([even.shape[0]] + [1 for _ in range(len(oldshape) - 1 - i)])
        idxs.append(even.reshape(paddedshape))
    return arr[*idxs]

print('loading data')

ts = OpenPMDTimeSeries(sys.argv[1])

print('loaded data')

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')


vals, griddata = ts.get_field(field='E', coord='x', iteration=0)
dimx, dimy, dimz = vals.shape
xgr, ygr, zgr = np.meshgrid(griddata.x, griddata.y, griddata.z, indexing='ij')

print(vals.shape)

print(len(griddata.x))
print(len(griddata.y))
print(len(griddata.z))

quiver_shape = (10, 10, 2)

xgr, ygr, zgr = trimshape(xgr, quiver_shape), trimshape(ygr, quiver_shape), trimshape(zgr, quiver_shape)

print('metadata loaded')

fact = 1e-2

for i in range(1):
    it = ts.iterations[i]
    u, _ = ts.get_field(field='B', coord='x', iteration=it)
    v, _ = ts.get_field(field='B', coord='y', iteration=it)
    w, _ = ts.get_field(field='B', coord='z', iteration=it)
    mags = np.sqrt(np.square(u) + np.square(v) + np.square(w))
    facts = np.log(mags + 1) / mags
    def postprocess(arr):
        arr = trimshape(arr.T, quiver_shape)
        arr *= fact
        return arr
    u = postprocess(u)
    v = postprocess(v)
    w = postprocess(w)
    ax.quiver(zgr, xgr, ygr, w, u, v, length=1, normalize=False)
    plt.show()
