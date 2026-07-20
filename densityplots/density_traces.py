from matplotlib import pyplot as plt
import numpy as np
import sys
from openpmd_viewer import OpenPMDTimeSeries
import cv2

path = sys.argv[1]
ts = OpenPMDTimeSeries(path)

width = 250
height = int(2.4 * width)

step_size = 100003

iters = ts.iterations
y, z = ts.get_particle(species = "beam1", var_list = ["x", "z"], iteration = 0)
y2, z2 = ts.get_particle(species = "beam2", var_list = ["x", "z"], iteration = 0)
y = np.array(y)
z = np.array(z)
y2 = np.array(y2)
z2 = np.array(z2)
miny = min(np.min(y), np.min(y2))
maxy = max(np.max(y), np.max(y2))
minz = min(np.min(z), np.min(z2))
maxz = max(np.max(z), np.max(z2))
diffz = maxz - minz
diffy = maxy - miny
offsetz = -0.05
offsety = -0.15
miny -= offsety * diffy
maxy += offsety * diffy
minz -= offsetz * diffz
maxz += offsetz * diffz

pho1 = ts.get_particle(species='pho1', var_list=['id'], iteration=400)[0]
pho1ids = np.array(pho1)[np.linspace(0, len(pho1)-1, 30, dtype=np.int32)]
pho2 = ts.get_particle(species='pho2', var_list=['id'], iteration=400)[0]
pho2ids = np.array(pho2)[np.linspace(0, len(pho2)-1, 30, dtype=np.int32)]

fig, ax = plt.subplots()

ax.set_xlim(minz, maxz)
ax.set_ylim(miny, maxy)
ax.set_aspect(2.4 * (maxz - minz) / (maxy - miny))

im = None

scatter = ax.scatter([], [], s=4, color='green')

plt.ion()
plt.show()

plt.pause(0.1)

vidname = ''
try:
    vidname = sys.argv[2]
except IndexError:
    pass

start = False

def clicked(event):
    global start
    if event.button == 3:
        start = True

fig.canvas.mpl_connect('button_press_event', clicked)

while not start:
    plt.pause(0.1)

id_scatter = None

for it in iters:
    yp, zp, ids1 = ts.get_particle(species = "beam1", var_list = ["x", "z", 'id'], iteration = it)
    y2p, z2p, ids2 = ts.get_particle(species = "beam2", var_list = ["x", "z", 'id'], iteration = it)
    ids1 = np.array(ids1) 
    ids2 = np.array(ids2)
    ids1_idx = np.where(ids1 % step_size == 0)[0]
    ids2_idx = np.where(ids2 % step_size == 0)[0]
    yp = np.array(yp)
    zp = np.array(zp)
    y2p = np.array(y2p)
    z2p = np.array(z2p)
    y = np.astype((yp-miny)/(maxy-miny) * height, np.int32)
    z = np.astype((zp-minz)/(maxz-minz) * width, np.int32)
    y2 = np.astype((y2p-miny)/(maxy-miny) * height, np.int32)
    z2 = np.astype((z2p-minz)/(maxz-minz) * width, np.int32)
    i1 = y * width + z
    i2 = y2 * width + z2
    i1[i1 >= width * height] = 0
    i1[i1 < 0] = 0
    i2[i2 >= width * height] = 0
    i2[i2 < 0] = 0
    bin1 = np.bincount(i1, minlength = width * height)
    bin2 = np.bincount(i2, minlength = width * height)
    densities = np.reshape(bin2-bin1, shape = (height, width))

    #img = np.random.random((height, width, 4))
    img = np.zeros(shape=(height, width, 4))
    abscount = np.abs(densities)
    maxcount = np.max(abscount)
    img[:, :, 3] = (abscount / maxcount) * 0.4
    red = (np.sign(densities) + 1) // 2
    blue = 1 - red
    img[:, :, 0] = red
    img[:, :, 2] = blue
    img = img[::-1]
    im_was_none = False
    if im is None:
        im_was_none = True
        im = ax.imshow(img, extent=(minz, maxz, miny, maxy), aspect='auto')#, interpolation='bilinear')
        linedict1 = {}
        linedict2 = {}
        for id1idx in ids1_idx:
            curid = ids1[id1idx]
            linedict1[curid] = [[zp[id1idx]], [yp[id1idx]], ax.plot([], [], color=(0, 0, 0, 0.2), linewidth=1)[0]]
        for id2idx in ids2_idx:
            curid = ids2[id2idx]
            linedict2[curid] = [[z2p[id2idx]], [y2p[id2idx]], ax.plot([], [], color=(0, 0, 0, 0.2), linewidth=1)[0]]
        zscat1 = zp[ids1_idx]
        zscat2 = z2p[ids2_idx]
        yscat1 = yp[ids1_idx]
        yscat2 = y2p[ids2_idx]
        id_scatter1 = ax.scatter(zscat1, yscat1, color='blue', s=36, edgecolors='black')
        id_scatter2 = ax.scatter(zscat2, yscat2, color='red', s=36, edgecolors='black')
    else:
        im.set_data(img)
        zscat1 = zp[ids1_idx]
        zscat2 = z2p[ids2_idx]
        yscat1 = yp[ids1_idx]
        yscat2 = y2p[ids2_idx]
        pts1 = np.array([zscat1, yscat1]).T
        pts2 = np.array([zscat2, yscat2]).T
        id_scatter1.set_offsets(pts1)
        id_scatter2.set_offsets(pts2)
        for id1idx in ids1_idx:
            curid = ids1[id1idx]
            if curid in linedict1:
                cur = linedict1[curid]
                cur[2].set_data(cur[0], cur[1])
                cur[0].append(zp[id1idx])
                cur[1].append(yp[id1idx])
        for id2idx in ids2_idx:
            curid = ids2[id2idx]
            if curid in linedict2:
                cur = linedict2[curid]
                cur[2].set_data(cur[0], cur[1])
                cur[0].append(z2p[id2idx])
                cur[1].append(y2p[id2idx])

    pho1 = ts.get_particle(species='pho1', var_list=['z', 'x', 'id'], iteration=it)
    pho2 = ts.get_particle(species='pho2', var_list=['z', 'x', 'id'], iteration=it)
    mask1 = np.isin(np.array(pho1[2]), pho1ids)
    mask2 = np.isin(np.array(pho2[2]), pho2ids)

    pho1 = (np.array(pho1).T)[mask1]
    pho2 = (np.array(pho2).T)[mask2]

    combined = np.append(pho1, pho2, axis=0)

    scatter.set_offsets(combined[:, :2])

    fig.canvas.draw_idle()

    print(f'finished iteration {it}')

    plt.pause(0.04)

    if vidname:
        if im_was_none:
            frame = np.asarray(fig.canvas.buffer_rgba())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            h, w, _ = frame.shape
            video = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)


if vidname:
    video.release()
