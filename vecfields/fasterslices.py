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

    iters = np.arange(0, 501, 4, dtype=np.int32)
    n_iters = len(iters)

    p = lambda s: sys.argv[3] + '/' + s + '.npy'
    np.save(p('xgr'), xgr)
    np.save(p('ygr'), ygr)
    np.save(p('iters'), iters)

    subsets = np.empty(shape=(n_iters, num_pts, 3))
    zdiffs = np.empty(shape=(n_iters, num_pts))
    zranges = np.empty(shape=(n_iters,))
    cols = np.empty(shape=(n_iters, num_pts, 3))
    us = np.empty(shape=(n_iters, *quiver_shape))
    vs = np.empty(shape=us.shape)
    mags = np.empty(shape=us.shape)

    ids1 = set()
    ids2 = set()
    sets1 = [None for _ in range(n_iters)]
    sets2 = [None for _ in range(n_iters)]
    i = 0
    for it in iters:
        ptsit = round(it/5) * 5
        sets1[i] = set(pts.get_particle(species='beam1', var_list=['id'], iteration=ptsit)[0])
        sets2[i] = set(pts.get_particle(species='beam2', var_list=['id'], iteration=ptsit)[0])
        print(f'iterated {it} out of {iters[-1]} for set creation')
        i += 1
    lastiter = round(iters[-1]/5) * 5
    idlist1 = pts.get_particle(species='beam1', var_list=['id'], iteration=lastiter)[0]
    idlist2 = pts.get_particle(species='beam2', var_list=['id'], iteration=lastiter)[0]
    def validateid(idx, idlist, setlist):
        while True:
            curid = idlist[idx]
            worked = True
            for i in range(n_iters):
                if not (curid in setlist[i]):
                    worked = False
                    print(f'{idx} had id {curid} and failed at iteration {i}')
                    break
            if worked:
                return curid
            else:
                idx += 1
    for i in np.linspace(0, len(idlist1), num_pts//2+2, dtype=np.int32)[1:-1]:
        ids1.add(validateid(i, idlist1, sets1))
    for i in np.linspace(0, len(idlist2), num_pts//2+2, dtype=np.int32)[1:-1]:
        ids2.add(validateid(i, idlist2, sets2))

    #print(ids1)
    #print(ids2)

    ids1 = np.array(list(ids1))
    ids2 = np.array(list(ids2))

    #def get_idxs(idarr, idset):
    #    res = np.empty(shape=(len(idset),), dtype=np.int32)
    #    j = 0
    #    for i in range(len(idarr)):
    #        if idarr[i] in idset:
    #            res[j] = i
    #            j += 1
    #    return res

    def get_mask(idarr, idset):
        mask = np.isin(idarr, idset)
        mask.reshape((mask.shape[0], 1))
        return mask

    i = 0
    for it in iters:
        vecsit = it
        ptsit = round(it/5) * 5
        curpts1 = np.array(pts.get_particle(species='beam1', var_list=['x', 'y', 'z'], iteration=ptsit))
        curpts2 = np.array(pts.get_particle(species='beam2', var_list=['x', 'y', 'z'], iteration=ptsit))
        curpts1 = np.append(curpts1, np.zeros(shape=(1, curpts1.shape[1])), axis=0).T
        curpts2 = np.append(curpts2, np.ones(shape=(1, curpts2.shape[1])), axis=0).T
        mask1 = get_mask(pts.get_particle(species='beam1', var_list=['id'], iteration=ptsit)[0], ids1)
        mask2 = get_mask(pts.get_particle(species='beam2', var_list=['id'], iteration=ptsit)[0], ids2)
        curpts1 = curpts1[mask1]
        curpts2 = curpts2[mask2]
        curpts = np.append(curpts1, curpts2, axis=0)
        idxs = np.argsort(curpts1[:, 2])
        curpts1 = curpts1[idxs]
        mididx = curpts1.shape[0]//2
        midz = curpts1[mididx, 2]
        zrange = ((curpts1[mididx + num_strong//2, 2] - midz) + (midz - curpts1[mididx - num_strong//2, 2])) / 2
        zdiff = curpts[:, 2] - midz
        subset = curpts[:, :3]
        col = np.zeros(shape=(num_pts, 3))
        col[:, 0] = curpts[:, 3]
        col[:, 2] = 1 - curpts[:, 3]
        zi = zidx(midz)
        u = trimshape(vecs.get_field(field='E', coord='x', iteration=vecsit)[0][zi], quiver_shape)
        v = trimshape(vecs.get_field(field='E', coord='y', iteration=vecsit)[0][zi], quiver_shape)
        mag = np.sqrt(np.square(u) + np.square(v))
        subsets[i] = subset
        zdiffs[i] = zdiff
        zranges[i] = zrange
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
    np.save(p('zdiffs'), zdiffs)
    np.save(p('zranges'), zranges)
    np.save(p('cols'), cols)
    np.save(p('us'), us)
    np.save(p('vs'), vs)
    np.save(p('mags'), mags)

    print(f'saved all values in folder {sys.argv[3]}')

    vididx = 4

else:

    print('loading saved data')

    load = lambda s: np.load(sys.argv[2] + '/' + s + '.npy')

    xgr = load('xgr')
    ygr = load('ygr')
    subsets = load('subsets')
    zdiffs = load('zdiffs')
    zranges = load('zranges')
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
    if vidname == 'no_display':
        print('exiting without animation')
        sys.exit()
    else:
        print(f'saving video as {vidname}')
else:
    print('proceding without saving video')

fig, ax = plt.subplots()

scatterobj = None
quiverobj = None

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
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

k = 0.05
i = 0
minlim, maxlim = mags.min(), mags.max()
for it in iters:
    u, v, mag, col, subset, zdiff, zrange = us[i], vs[i], mags[i], cols[i], subsets[i], zdiffs[i], zranges[i]
    f = np.sqrt(np.log(1/k)/zrange) * 6
    decayval = np.exp(-np.square(zdiff * f))
    color = np.append(col, decayval.reshape((col.shape[0], 1)), axis=1)
    if it == 0:
        quiverobj = ax.quiver(xgr, ygr, u, v, mag, cmap='viridis')
        quiverobj.set_clim(minlim, maxlim)
        cbar = fig.colorbar(quiverobj, ax=ax)
        scatterobj = ax.scatter(subset[:, 0], subset[:, 1], c=color, s=decayval*5)
    else:
        scatterobj.set_offsets(subset)
        scatterobj.set_facecolors(color)
        scatterobj.set_sizes(decayval*5)
        quiverobj.set_UVC(u, v, mag)
    fig.canvas.draw_idle()
    plt.pause(0.01)
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    print(f'finished drawing frame {i+1} out of {len(iters)}')
    i += 1

print('done')
