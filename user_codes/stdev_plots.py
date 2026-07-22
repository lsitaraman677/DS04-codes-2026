from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import sys
import numpy as np

loaded = False
if sys.argv[-2]== 'load':
    names = [sys.argv[i] for i in range(3, len(sys.argv)-2, 2)]
    paths = [sys.argv[i] for i in range(4, len(sys.argv)-2, 2)]
    beam1 = sys.argv[1]
    beam2 = sys.argv[2]
    data = np.load(sys.argv[-1])
    n = data.shape[2]
    t = []
    dat = OpenPMDTimeSeries(paths[0])
    for i in range(n):
        it_idx = int((i/n) * len(dat.iterations))
        t.append(dat.t[it_idx])
    loaded = True
else:
    names = [sys.argv[i] for i in range(3, len(sys.argv), 2)]
    paths = [sys.argv[i] for i in range(4, len(sys.argv), 2)]
    beam1 = sys.argv[1]
    beam2 = sys.argv[2]

if not loaded:
    datas = [OpenPMDTimeSeries(i) for i in paths]

    t = []
    n = 100
    data = np.empty(shape=(len(paths), 2, n, 3))
    for i in range(n):
        for j in range(len(datas)):
            it_idx = int((i/n) * len(datas[j].iterations))
            it = datas[j].iterations[it_idx]
            if j == 0:
                t.append(datas[j].t[it_idx])
            x, y, z = datas[j].get_particle(species=beam1, var_list=['x', 'y', 'z'], iteration=it)
            data[j, 0, i, 0] = np.std(np.array(x))
            data[j, 0, i, 1] = np.std(np.array(y))
            data[j, 0, i, 2] = np.std(np.array(z))
            x, y, z = datas[j].get_particle(species=beam2, var_list=['x', 'y', 'z'], iteration=it)
            data[j, 1, i, 0] = np.std(np.array(x))
            data[j, 1, i, 1] = np.std(np.array(y))
            data[j, 1, i, 2] = np.std(np.array(z))
            print(f'finished {paths[j]} for iteration {i}')

    np.save('mustsave.npy', data)
    data /= data[:, :, [0], :]

    print(f'done collecting data')

fig, ax = plt.subplots(2, 3, layout='constrained')

for beam in range(2):
    for coord in range(3):
        for j in range(len(names)):
            if beam == 0 and coord == 0:
                ax[beam][coord].plot(t, data[j, beam, :, coord], label=names[j])
            else:
                ax[beam][coord].plot(t, data[j, beam, :, coord])
        ax[beam][coord].set_title(f'{[beam1, beam2][beam]}: {['x', 'y', 'z'][coord]} stdev')
        ax[beam][coord].set_xlabel('time')
        ax[beam][coord].set_ylabel('stdev (normalized)')

fig.legend(loc="lower right")

ax[0][1].set_ylim(0, 6)
ax[1][1].set_ylim(0, 6)

plt.ion()
plt.show()

plt.pause(5.0)

fig.canvas.draw_idle()

plt.savefig('stdev_plot.png')


