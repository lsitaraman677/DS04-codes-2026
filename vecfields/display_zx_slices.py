from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
import sys
import matplotlib

print('loading saved data')

load = lambda s: np.load(sys.argv[1] + '/' + s + '.npy')

xgr = load('xgr')
zgr = load('zgr')
subsets = load('subsets')
ydiffs = load('ydiffs')
yranges = load('yranges')
cols = load('cols')
ubs = load('ubs')
vbs = load('vbs')
ues = load('ues')
ves = load('ves')
iters = load('iters')

xdiff = xgr[-1, -1] - xgr[0, 0]
xmin, xmax = xgr[0, 0] - 0.05 * xdiff, xgr[-1, -1] + 0.05 * xdiff
zdiff = zgr[-1, -1] - zgr[0, 0]
zmin, zmax = zgr[0, 0] - 0.05 * zdiff, zgr[-1, -1] + 0.05 * zdiff

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
im = None

ax.set_xlim(zmin, zmax)
ax.set_ylim(xmin, xmax)
ax.set_xlabel('z (m)')
ax.set_ylabel('x (m)')
#ax.set_aspect('equal')
ax.set_box_aspect(quiver_shape[0] / quiver_shape[1])

legend_elements = [
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Beam 1 (electrons)'),
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Beam 2 (positrons)')
]
ax.legend(handles=legend_elements)

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
    video = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height))

k = 0.05
i = 0
emags = np.sqrt(np.square(ues) + np.square(ves))
bmags = np.sqrt(np.square(ubs) + np.square(vbs))
extent = (zgr[0, 0], zgr[-1, -1], xgr[0, 0], xgr[-1, -1])
for it in iters:
    ue, ve, emag, ub, vb, bmag, col, subset, ydiff, yrange = ues[i], ves[i], emags[i], ubs[i], vbs[i], bmags[i], cols[i], subsets[i], ydiffs[i], yranges[i]
    f = np.sqrt(np.log(1/k)) / ydiff
    decayval = np.exp(-np.square(ydiff * f))
    color = np.append(col, decayval.reshape((col.shape[0], 1)), axis=1)
    if it == 0:
        im = ax.imshow(bmag, extent=extent, aspect='auto', cmap='cool', alpha=0.2, interpolation='bilinear')
        ibar = fig.colorbar(im, ax=ax, location='left', label='Magnetic Field Strength (V/m)')
        quiverobj = ax.quiver(zgr, xgr, ue, ve, emag, cmap='plasma')
        qbar = fig.colorbar(quiverobj, ax=ax, location='right', label='Electric Field Strength (V/m)')
        scatterobj = ax.scatter(subset[:, 2], subset[:, 0], c=color, s=decayval*5)
    else:
        scatterobj.set_offsets(subset[:, [2, 0]])
        scatterobj.set_facecolors(color)
        scatterobj.set_sizes(decayval*5)
        quiverobj.set_UVC(ue, ve, emag)
        #quiverobj.set_clim(emag.min(), emag.max())
        im.set_data(bmag)
    fig.canvas.draw_idle()
    plt.pause(0.01)
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    print(f'finished drawing frame {i+1} out of {len(iters)}')
    i += 1

print('done')
