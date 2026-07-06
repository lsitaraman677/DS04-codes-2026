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

vididx = None

if sys.argv[1] != 'load':

    print('loading data')

    vecs = OpenPMDTimeSeries(sys.argv[1])
    pts = OpenPMDTimeSeries(sys.argv[2])

    print(vecs.iterations)
    print(pts.iterations)

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

    startz = griddata.z[0]
    endz = griddata.z[-1]
    lenz = len(griddata.z)
    zidx = lambda z: int(((z - startz) / (endz - startz)) * lenz)

    num_pts = 1000
    num_strong = 50
    k = 0.05

    iters = np.arange(0, 501, 4, dtype=np.int32)
    n_iters = len(iters)

    p = lambda s: sys.argv[3] + '/' + s + '.npy'
    np.save(p('xgr'), xgr)
    np.save(p('ygr'), ygr)
    np.save(p('iters'), iters)

    subsets = np.empty(shape=(n_iters, num_pts, 3))
    cols = np.empty(shape=(n_iters, num_pts, 4))
    us = np.empty(shape=(n_iters, *quiver_shape))
    vs = np.empty(shape=us.shape)
    mags = np.empty(shape=us.shape)

    i = 0
    for it in iters:
        vecsit = it
        ptsit = round(it/5) * 5
        curpts = np.array(pts.get_particle(species='beam1', var_list=['x', 'y', 'z'], iteration=ptsit)).T
        idxs = np.argsort(curpts[:, 2])
        curpts = curpts[idxs]
        mididx = curpts.shape[0]//2
        midz = curpts[mididx, 2]
        zdiff = ((curpts[mididx + num_strong//2, 2] - midz) + (midz - curpts[mididx - num_strong//2, 2])) / 2
        f = np.sqrt(np.log(1/k)/zdiff)
        subset = curpts[np.linspace(0, len(curpts)-1, num_pts, dtype=np.int32)]
        avals = np.exp(-np.square((subset[:, 2] - midz) * f))
        col = (avals + np.zeros((4, 1))).T
        col[:, 0] = 0.0
        col[:, 1] = 0.0
        col[:, 2] = 1.0
        zi = zidx(midz)
        u = trimshape(vecs.get_field(field='E', coord='x', iteration=vecsit)[0][zi], quiver_shape)
        v = trimshape(vecs.get_field(field='E', coord='y', iteration=vecsit)[0][zi], quiver_shape)
        mag = np.sqrt(np.square(u) + np.square(v))
        subsets[i] = subset
        cols[i] = col
        us[i] = u
        vs[i] = v
        mags[i] = mag
        print(f'finished iteration {it} out of {iters[-1]}')
        i += 1

    xdiff = griddata.x[-1] - griddata.x[0]
    xmin, xmax = griddata.x[0] - 0.05 * xdiff, griddata.x[-1] + 0.05 * xdiff
    ydiff = griddata.y[-1] - griddata.y[0]
    ymin, ymax = griddata.y[0] - 0.05 * ydiff, griddata.y[-1] + 0.05 * ydiff

    np.save(p('subsets'), subsets)
    np.save(p('cols'), cols)
    np.save(p('us'), us)
    np.save(p('vs'), vs)
    np.save(p('mags'), mags)

    print(f'saved all values in folder {sys.argv[3]}')

    vididx = 4

else:

    print('loading saved data')

    load = lambda s: np.load(sys.argv[2] + s + '.npy')

    xgr = load('xgr')
    ygr = load('ygr')
    subsets = load('subsets')
    cols = load('cols')
    us = load('us')
    vs = load('vs')
    mags = load('mags')
    iters = load('iters')

    xdiff = xgr[-1, -1] - xgr[0, 0]
    xmin, xmax = xgr[0, 0] - 0.05 * xdiff, xgr[-1, -1] + 0.05 * xdiff
    ydiff = ygr[-1, -1] - ygr[0, 0]
    ymin, ymax = ygr[0, 0] - 0.05 * ydiff, ygr[-1, -1] + 0.05 * ydiff

    quiver_shape = xgr.shape

    print('loaded saved data')

    vididx = 3



import cv2

vidname = ''
try:
    vidname = sys.argv[vididx]
except:
    pass
if vidname:
    print(f'saving video as {vidname}')
else:
    print('proceding without saving video')

fig, ax = plt.subplots()

scatterobj = None
quiverobj = None

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
#ax.set_aspect('equal')
ax.set_box_aspect(quiver_shape[0] / quiver_shape[1])

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

i = 0
for it in iters:
    u, v, mag, col, subset = us[i], vs[i], mags[i], cols[i], subsets[i]
    if it == 0:
        quiverobj = ax.quiver(xgr, ygr, u, v, mag, cmap='viridis')
        scatterobj = ax.scatter(subset[:, 0], subset[:, 1], c=col)
    else:
        scatterobj.set_offsets(subset)
        scatterobj.set_facecolors(col)
        quiverobj.set_UVC(u, v)
    fig.canvas.draw_idle()
    plt.pause(0.01)
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    print(f'finished drawing frame {i+1} out of {len(iters)}')
    i += 1

print('done')
