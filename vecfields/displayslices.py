from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
import sys


print('loading saved data')

load = lambda s: np.load(sys.argv[1] + '/' + s + '.npy')

xgr = load('xgr')
ygr = load('ygr')
subsets = load('subsets')
zdiffs = load('zdiffs')
zranges = load('zranges')
cols = load('cols')
ubs = load('ubs')
vbs = load('vbs')
ues = load('ues')
ves = load('ves')
iters = load('iters')

xdiff = xgr[-1, -1] - xgr[0, 0]
xmin, xmax = xgr[0, 0] - 0.05 * xdiff, xgr[-1, -1] + 0.05 * xdiff
ydiff = ygr[-1, -1] - ygr[0, 0]
ymin, ymax = ygr[0, 0] - 0.05 * ydiff, ygr[-1, -1] + 0.05 * ydiff

quiver_shape = xgr.shape

print('loaded saved data')

import cv2

vidname = ''
try:
    vidname = sys.argv[2]
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
emags = np.sqrt(np.square(ues) + np.square(ves))
bmags = np.sqrt(np.square(ubs) + np.square(vbs))
for it in iters:
    ue, ve, emag, ub, vb, bmag, col, subset, zdiff, zrange = ues[i], ves[i], emags[i], ubs[i], vbs[i], bmags[i], cols[i], subsets[i], zdiffs[i], zranges[i]
    f = np.sqrt(np.log(1/k)/zrange) * 6
    decayval = np.exp(-np.square(zdiff * f))
    color = np.append(col, decayval.reshape((col.shape[0], 1)), axis=1)
    if it == 0:
        quiverobj = ax.quiver(xgr, ygr, ue, ve, emag, cmap='viridis')
        cbar = fig.colorbar(quiverobj, ax=ax)
        scatterobj = ax.scatter(subset[:, 0], subset[:, 1], c=color, s=decayval*5)
    else:
        scatterobj.set_offsets(subset)
        scatterobj.set_facecolors(color)
        scatterobj.set_sizes(decayval*5)
        quiverobj.set_UVC(u, v, emag)
        quiverobj.set_clim(emag.min(), emag.max())
    fig.canvas.draw_idle()
    plt.pause(0.01)
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    print(f'finished drawing frame {i+1} out of {len(iters)}')
    i += 1

print('done')
