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

vecs = OpenPMDTimeSeries(sys.argv[1])
pts = OpenPMDTimeSeries(sys.argv[2])

print('loaded data')

vals, griddata = vecs.get_field(field='E', coord='x', iteration=0)
dimx, dimy, dimz = vals.shape
#xgr, ygr, zgr = np.meshgrid(griddata.x, griddata.y, griddata.z, indexing='ij')

print(vals.shape)

print(len(griddata.x))
print(len(griddata.y))
print(len(griddata.z))

quiver_shape = (16, 12)

xgr = trimshape(griddata.x, (quiver_shape[1],))
ygr = trimshape(griddata.y, (quiver_shape[0],))

xgr, ygr = np.meshgrid(xgr, ygr, indexing='xy')

import cv2
vidname = input('enter video name: ')

fig, ax = plt.subplots()

scatterobj = None
quiverobj = None

xdiff = griddata.x[-1] - griddata.x[0]
xmin, xmax = griddata.x[0] - 0.05 * xdiff, griddata.x[-1] + 0.05 * xdiff
ydiff = griddata.y[-1] - griddata.y[0]
ymin, ymax = griddata.y[0] - 0.05 * ydiff, griddata.y[-1] + 0.05 * ydiff
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
#ax.set_aspect('equal')
ax.set_box_aspect(quiver_shape[0] / quiver_shape[1])

print(vecs.iterations)
print(pts.iterations)

plt.ion()
plt.show()

start = False
def clicked(event):
    global start
    if event.button == 2:
        start = True

fig.canvas.mpl_connect('button_press_event', clicked)

while not start:
    plt.pause(0.01)

print('starting animation')

if vidname:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    height, width, _ = frame.shape
    video = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

startz = griddata.z[0]
endz = griddata.z[-1]
lenz = len(griddata.z)
zidx = lambda z: int(((z - startz) / (endz - startz)) * lenz)

num_pts = 1000
num_strong = 50
k = 0.05

for it in range(0, 501, 4):
    vecsit = it
    ptsit = round(it/5) * 5
    curpts = np.array(pts.get_particle(species='beam1', var_list=['x', 'y', 'z'], iteration=ptsit)).T
    idxs = np.argsort(curpts[:, 2])
    curpts = curpts[idxs]
    mididx = curpts.shape[0]//2
    midz = curpts[mididx, 2]
    zdiff = ((curpts[mididx + num_strong//2, 2] - midz) + (midz - curpts[mididx - num_strong//2, 2])) / 2
    f = np.sqrt(np.log(1/k)/zdiff)
    subset = curpts[np.arange(0, len(curpts), (len(curpts) // num_pts), dtype=np.int32)]
    avals = np.exp(-np.square((subset[:, 2] - midz) * f))
    cols = (avals + np.zeros((4, 1))).T
    cols[:, 0] = 0.0
    cols[:, 1] = 0.0
    cols[:, 2] = 1.0
    zi = zidx(midz)
    u = trimshape(vecs.get_field(field='E', coord='x', iteration=vecsit)[0][zi], quiver_shape)
    v = trimshape(vecs.get_field(field='E', coord='y', iteration=vecsit)[0][zi], quiver_shape)
    mags = np.sqrt(np.square(u) + np.square(v))
    if it == 0:
        quiverobj = ax.quiver(xgr, ygr, u, v, mags, cmap='viridis')
        scatterobj = ax.scatter(subset[:, 0], subset[:, 1], c=cols)
        #ax.scatter(xgr.flatten(), ygr.flatten())
    else:
        scatterobj.set_offsets(subset)
        scatterobj.set_facecolors(cols)
        quiverobj.set_UVC(u, v)
    fig.canvas.draw_idle()
    plt.pause(0.01)
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    print(f'finished iteration {it}')

print('done')


