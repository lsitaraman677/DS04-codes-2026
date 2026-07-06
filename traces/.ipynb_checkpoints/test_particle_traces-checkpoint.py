from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import sys
import numpy as np

ts = OpenPMDTimeSeries(sys.argv[1])

n_particles = 10
newiters = np.array(ts.iterations)[np.linspace(0, len(ts.iterations)-1, 100, dtype=np.int32)]
n_iters = newiters.shape[0]

data1 = np.empty(shape=(n_iters, n_particles, 3))
data2 = np.empty(shape=(n_iters, n_particles, 3))

def tonumpy(pmddata, datsize=None):
    if datsize is None:
        datsize = len(pmddata[0])
    unfiltered = np.array(pmddata).T
    idxs = np.linspace(0, len(pmddata[0])-1, datsize, dtype=np.int32)
    return unfiltered[idxs]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot([], [], [])
plt.ion()
plt.show()
plt.pause(0.1)

for i in range(n_iters):
    beam1 = tonumpy(ts.get_particle(species='beam1', var_list=['x', 'y', 'z'], iteration=newiters[i]), datsize=n_particles)
    beam2 = tonumpy(ts.get_particle(species='beam2', var_list=['x', 'y', 'z'], iteration=newiters[i]), datsize=n_particles)
    data1[i] = beam1
    data2[i] = beam2
    ax.clear()
    ax.scatter(beam1[:, 0], beam1[:, 1], beam1[:, 2], c='blue', s=10)
    ax.scatter(beam2[:, 0], beam2[:, 1], beam2[:, 2], c='red', s=10)
    if i > 1:
        for j in range(n_particles):
            for data in (data1, data2):
                trace = data[:i, j, :]
                ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], 'black')
    fig.canvas.draw_idle()
    plt.pause(0.1)
    print(f'finished {i} iterations')

#ax.clear()
points = data1[:, 1, :]
#ax.plot(points[:, 0], points[:, 1], points[:, 2])

plt.close('all')
plt.plot(np.arange(points.shape[0]), points[:, 0])

plt.show()
