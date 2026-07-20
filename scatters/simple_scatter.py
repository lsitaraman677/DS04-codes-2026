from matplotlib import pyplot as plt
import numpy as np
import sys
from openpmd_viewer import OpenPMDTimeSeries
import cv2

path = sys.argv[1]
ts = OpenPMDTimeSeries(path)

iters = ts.iterations
y, z = ts.get_particle(species = "beam1", var_list = ["y", "z"], iteration = 400)
y2, z2 = ts.get_particle(species = "beam2", var_list = ["y", "z"], iteration = 400)
y3, z3 = ts.get_particle(species = "pho1", var_list = ["y", "z"], iteration = 400)
y4, z4 = ts.get_particle(species = "pho2", var_list = ["y", "z"], iteration = 400)
y3 = np.append(np.array(y3), np.array(y4))
z3 = np.append(np.array(z3), np.array(z4))

y = np.array(y)
z = np.array(z)
y2 = np.array(y2)
z2 = np.array(z2)
miny = min(np.min(y), np.min(y2), np.min(y3))
maxy = max(np.max(y), np.max(y2), np.max(y3))
minz = min(np.min(z), np.min(z2), np.min(z3))
maxz = max(np.max(z), np.max(z2), np.max(z3))

diffz = maxz - minz
diffy = maxy - miny
offsetz = 0.1
offsety = 0.1
miny -= offsety * diffy
maxy += offsety * diffy
minz -= offsetz * diffz
maxz += offsetz * diffz

im = None

fig, ax = plt.subplots()
ax.set_xlim(minz, maxz)
ax.set_ylim(miny, maxy)
ax.set_xlabel('z')
ax.set_ylabel('y')
ax.set_title('Z vs Y, positrons, electrons, and emitted photons')

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

scat1 = ax.scatter([], [], color='blue', s=1)
scat2 = ax.scatter([], [], color='red',  s=1)
scat3 = ax.scatter([], [], color='green', s=1)

for it in iters:
    y, z = ts.get_particle(species = "beam1", var_list = ["y", "z"], iteration = it)
    y2, z2 = ts.get_particle(species = "beam2", var_list = ["y", "z"], iteration = it)
    y3, z3 = ts.get_particle(species = "pho1", var_list = ["y", "z"], iteration = it)
    y4, z4 = ts.get_particle(species = "pho2", var_list = ["y", "z"], iteration = it)
    y3 = np.append(np.array(y3), np.array(y4))
    z3 = np.append(np.array(z3), np.array(z4))

    scat1.remove()
    scat1 = ax.scatter(z, y, color='blue', s=1)

    scat2.remove()
    scat2 = ax.scatter(z2, y2, color='red', s=1)

    scat3.remove()
    scat3 = ax.scatter(z3, y3, color='green', s=1)

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
