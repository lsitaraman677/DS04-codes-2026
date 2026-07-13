from matplotlib import pyplot as plt
import numpy as np
import sys
from openpmd_viewer import OpenPMDTimeSeries
import cv2

path = sys.argv[1]
ts = OpenPMDTimeSeries(path)

width = 500
height = int(2.4 * width)

iters = ts.iterations
x, z = ts.get_particle(species = "beam1", var_list = ["x", "z"], iteration = 0)
x2, z2 = ts.get_particle(species = "beam2", var_list = ["x", "z"], iteration = 0)
x = np.array(x)
z = np.array(z)
x2 = np.array(x2)
z2 = np.array(z2)
minx = min(np.min(x), np.min(x2))
maxx = max(np.max(x), np.max(x2)) 
minz = min(np.min(z), np.min(z2))
maxz = max(np.max(z), np.max(z2))
diffz = maxz - minz
diffx = maxx - minx
offsetz = 0
offsetx = 0
minx -= offsetx * diffx
maxx += offsetx * diffx
minz -= offsetz * diffz
maxz += offsetz * diffz

fig, ax = plt.subplots()

ax.set_xlim(minz, maxz)
ax.set_xlim(minx, maxx)
ax.set_aspect(2.4 * (maxz - minz) / (maxx - minx))

im = None

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
    plt.pause(0.01)

for it in iters:
    x, z = ts.get_particle(species = "beam1", var_list = ["x", "z"], iteration = it)
    x2, z2 = ts.get_particle(species = "beam2", var_list = ["x", "z"], iteration = it)
    x = np.array(x)
    z = np.array(z)
    x2 = np.array(x2)
    z2 = np.array(z2)
    x = np.astype((x-minx)/(maxx-minx) * height, np.int32)
    z = np.astype((z-minz)/(maxz-minz) * width, np.int32)
    x2 = np.astype((x2-minx)/(maxx-minx) * height, np.int32)
    z2 = np.astype((z2-minz)/(maxz-minz) * width, np.int32)
    i1 = x * width + z
    i2 = x2 * width + z2
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
    img[:, :, 3] = (abscount / maxcount)
    red = (np.sign(densities) + 1) // 2
    blue = 1 - red
    img[:, :, 0] = red
    img[:, :, 2] = blue
    img = img[::-1]

    im_was_none = False
    if im is None:
        im_was_none = True
        im = ax.imshow(img, extent=(minz, maxz, minx, maxx), aspect='auto')#, interpolation='bilinear')
    else:
        im.set_data(img)

    fig.canvas.draw_idle()

    print(f'finished iteration {it}')

    plt.pause(0.01)

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

