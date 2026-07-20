from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
import sys
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

print('loading saved data')

load = lambda s: np.load(sys.argv[1] + '/' + s + '.npy')

xgr = load('xgr')
ygr = load('ygr')
subsets = load('subsets')
zdiffs = load('zdiffs')
zranges = load('zranges')
zmids = load('zmids')
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

minx, maxx = xgr[0, 0] + 0.05 * xdiff, xgr[-1, -1] - 0.05 * xdiff
miny, maxy = ygr[0, 0] + 0.05 * ydiff, ygr[-1, -1] - 0.05 * ydiff

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

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax3d = fig.add_subplot(1, 2, 2, projection='3d')

scatterobj = None
quiverobj = None
im = None

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#ax.set_aspect('equal')
ax.set_box_aspect(quiver_shape[0] / quiver_shape[1])

ax3d.set_ylim(xmin, xmax)
ax3d.set_zlim(ymin, ymax)
ax3d.set_xlim(-0.14, 0.14)
ax3d.set_box_aspect((3, 1, quiver_shape[0] / quiver_shape[1]))
ax3d.set_xlabel('z')
ax3d.set_ylabel('x')
ax3d.set_zlabel('y')

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
    plt.pause(0.05)

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
extent = (xgr[0, 0], xgr[-1, -1], ygr[0, 0], ygr[-1, -1])
for it in iters:
    ue, ve, emag, ub, vb, bmag, col, subset, zdiff, zrange, midz = ues[i], ves[i], emags[i], ubs[i], vbs[i], bmags[i], cols[i], subsets[i], zdiffs[i], zranges[i], zmids[i]
    f = np.sqrt(np.log(1/k)/zrange) * 6
    decayval = np.exp(-np.square(zdiff * f))
    color = np.append(col, decayval.reshape((col.shape[0], 1)), axis=1)
    if it == 0:
        #im = ax.imshow(bmag, extent=extent, aspect='auto', cmap='cool', alpha=0.2, interpolation='bilinear')
        #ibar = fig.colorbar(im, ax=ax, location='left', label='Magnetic Field Strength (V/m)')
        quiverobj = ax.quiver(xgr, ygr, ue, ve, emag, cmap='plasma')
        qbar = fig.colorbar(quiverobj, ax=ax, location='right', label='Electric Field Strength (V/m)')
        scatterobj = ax.scatter(subset[:, 0], subset[:, 1], c=color, s=decayval*5)
        scat3d = ax3d.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=color[:, :3], s=1)
        poly = [np.array([(midz, minx, miny), (midz, minx, maxy), (midz, maxx, maxy), (midz, maxx, miny)])]
        polyobj = Poly3DCollection(poly, facecolors=(0, 0, 0, 0.1), edgecolors='black')
        ax3d.add_collection3d(polyobj)
    else:
        scatterobj.set_offsets(subset)
        scatterobj.set_facecolors(color)
        scatterobj.set_sizes(decayval*5)
        quiverobj.set_UVC(ue, ve, emag)
        scat3d._offsets3d = [subset[:, 2], subset[:, 0], subset[:, 1]]
        poly[0][:, 0] = midz
        polyobj.set_verts(poly)
        #quiverobj.set_clim(emag.min(), emag.max())
        #im.set_data(bmag)
    fig.canvas.draw_idle()
    plt.pause(0.01)
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    print(f'finished drawing frame {i+1} out of {len(iters)}')
    i += 1

print('done')
