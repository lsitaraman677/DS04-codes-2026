from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import sys
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


loaded = False
try:
    fname = sys.argv[2]
    if fname == 'load':
        print('loading saved data')
        data1 = np.load(f'{sys.argv[3]}_beam1.npy')
        data2 = np.load(f'{sys.argv[3]}_beam2.npy')
        n_iters, n_particles, _ = data1.shape
        loaded = True
except IndexError:
    fname = None
    
if not loaded:

    n_iters = 100
    n_particles = 10

    ts = OpenPMDTimeSeries(sys.argv[1])

    data1 = np.full((n_iters, n_particles, 3), np.nan)
    data2 = np.full((n_iters, n_particles, 3), np.nan)

    iters = np.array(ts.iterations)[np.linspace(0, len(ts.iterations)-1, n_iters, dtype=np.int32)]

    def arr_to_idxdict(arr):
        d = {}
        for i in range(len(arr)):
            d[arr[i]] = i
        return d

    beam1_ids = np.array(ts.get_particle(species='beam1', var_list=['id'], iteration=0)[0])
    beam1_ids = arr_to_idxdict(beam1_ids[np.linspace(0, len(beam1_ids)-1, n_particles, dtype=np.int32)])

    beam2_ids = np.array(ts.get_particle(species='beam2', var_list=['id'], iteration=0)[0])
    beam2_ids = arr_to_idxdict(beam2_ids[np.linspace(0, len(beam2_ids)-1, n_particles, dtype=np.int32)])

    for i in range(n_iters):
        beam1 = np.array(ts.get_particle(species='beam1', var_list=['z', 'x', 'y', 'id'], iteration=iters[i])).T
        weee = [0 for _ in range(n_particles)]
        for j in range(beam1.shape[0]):
            if beam1[j, 3] in beam1_ids:
                data1[i, beam1_ids[beam1[j, 3]]] = beam1[j, :3]
                weee[beam1_ids[beam1[j, 3]]] += 1
        beam2 = np.array(ts.get_particle(species='beam2', var_list=['z', 'x', 'y', 'id'], iteration=iters[i])).T
        for j in range(beam2.shape[0]):
            if beam2[j, 3] in beam2_ids:
                data2[i, beam2_ids[beam2[j, 3]]] = beam2[j, :3]
                weee[beam2_ids[beam2[j, 3]]] += 1
        if 1 in weee or 0 in weee:
            print(weee)
        print(f'finished {i} iterations')

    interptimes = 5
    tarr = (np.arange(interptimes) / interptimes).reshape(interptimes, 1, 1)
    oneminus = 1 - tarr
    newdata1 = np.empty(shape=((n_iters-1) * interptimes, n_particles, 3))
    newdata2 = np.empty(shape=((n_iters-1) * interptimes, n_particles, 3)) 
    for i in range(n_iters-1):
        newdata1[i*interptimes:(i+1)*interptimes] = data1[i][np.newaxis] * oneminus + data1[i+1][np.newaxis] * tarr
        newdata2[i*interptimes:(i+1)*interptimes] = data2[i][np.newaxis] * oneminus + data2[i+1][np.newaxis] * tarr
    data1 = newdata1
    data2 = newdata2
    n_iters *= interptimes

    if not (fname is None):
        np.save(f'{fname}_beam1.npy', data1)
        np.save(f'{fname}_beam2.npy', data2)

print(data1.shape)
print(data2.shape)

import cv2
vidname = input('enter video name: ')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect((2, 1.5, 1))
#ax.axis('off')
x1, x2 = data1[:, :, 0], data2[:, :, 0]
y1, y2 = data1[:, :, 1], data2[:, :, 1]
z1, z2 = data1[:, :, 2], data2[:, :, 2]
minx, maxx = min(np.nanmin(x1), np.nanmin(x2)), max(np.nanmax(x1), np.nanmax(x2))
miny, maxy = min(np.nanmin(y1), np.nanmin(y2)), max(np.nanmax(y1), np.nanmax(y2))
minz, maxz = min(np.nanmin(z1), np.nanmin(z2)), max(np.nanmax(z1), np.nanmax(z2))

vecx, vecy, vecz = np.array([maxx - minx, 0, 0]), np.array([0, maxy - miny, 0]), np.array([0, 0, maxz - minz])
origin = np.array([minx, miny, minz])
def parapoints(start, v1, v2): return np.array([start, start+v1, start+v1+v2, start+v2])
faces = [parapoints(origin, vecx, vecy), parapoints(origin, vecx, vecz),
         parapoints(origin, vecy, vecz), parapoints(origin+vecz, vecx, vecy),
         parapoints(origin+vecy, vecx, vecz), parapoints(origin+vecx, vecy, vecz)]
poly = Poly3DCollection(faces, facecolors='gray', edgecolors='gray', alpha=0.01)
ax.add_collection3d(poly)
ax.set_axis_off()

#ax.set_xlim(minz*0.02, maxz*0.02)
#ax.set_ylim(minz*0.002, maxz*0.002)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_zlim(minz, maxz)

lines1 = [ax.plot([], [], [], c=(0.0, 0.0, 0.0, 0.5))[0] for _ in range(n_particles)]
lines2 = [ax.plot([], [], [], c=(0.0, 0.0, 0.0, 0.5))[0] for _ in range(n_particles)] # comment this out to get rid of line paths

dummyarr = lambda: [0 for _ in range(n_particles)]
scatter1 = ax.scatter(dummyarr(), dummyarr(), dummyarr(), c='blue', s=10)   #comment this out to only have one beam dots
scatter2 = ax.scatter(dummyarr(), dummyarr(), dummyarr(), c='red', s=10)

plt.ion()
plt.show()

start = False
def clicked(event):
    global start
    if event.button == 3:
        start = True

fig.canvas.mpl_connect('button_press_event', clicked)

while not start:
    plt.pause(0.01)

if vidname:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    height, width, _ = frame.shape
    video = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

start_e = 25
pers_e = 1
peroff_e = 0.25
amp_e = 25
start_a = -90
revs_a = 1

for i in range(n_iters):
    scatter1._offsets3d = data1[i, :].T  #also comment this out for beam dots
    scatter2._offsets3d = data2[i, :].T
    for j in range(n_particles):
        lines1[j].set_data_3d(data1[:i, j, 0], data1[:i, j, 1], data1[:i, j, 2])  #comment this out to get rid of beam line paths 
        lines2[j].set_data_3d(data2[:i, j, 0], data2[:i, j, 1], data2[:i, j, 2]) 
    e = start_e + np.sin(2*np.pi*(pers_e*(i / n_iters) + peroff_e)) * amp_e
    a = start_a + (i / n_iters) * 360 * revs_a
    ax.view_init(elev=e, azim=a)
    fig.canvas.draw_idle()
    plt.pause(0.01)
    print(f'done with {i} out of {n_iters}')
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)

if vidname:
    video.release()

