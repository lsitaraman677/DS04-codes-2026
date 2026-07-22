import sys
from openpmd_viewer import OpenPMDTimeSeries
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib 

dat = OpenPMDTimeSeries(sys.argv[1])

elvchecks = [(30, -60), (30, -45), (20, -45), (10, -45), (30, -30), (30, -20), (20, -20)]

elvchecks = [(20, -40)]

for elv, az in elvchecks:

    fig = plt.figure(figsize=(17, 5))
    axs = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]

    print(len(dat.iterations))
    iters = [0, 250, 500]
    iters = [dat.iterations[i] for i in iters]

    offsetx=0.1
    offsety=0.1
    offsetz=0.1

    x1, y1, z1 = dat.get_particle(species='test1', var_list=['x', 'y', 'z'], iteration=500)
    x2, y2, z2 = dat.get_particle(species='test2', var_list=['x', 'y', 'z'], iteration=500)

    mi = lambda a, b: min(min(a), min(b))
    ma = lambda a, b: max(max(a), max(b))
    minx, maxx = mi(x1, x2), ma(x1, x2)
    miny, maxy = mi(y1, y2), ma(y1, y2)
    minz, maxz = mi(z1, z2), ma(z1, z2)

    sx, bx, sy, by, sz, bz = [(0+i) for i in [minx, maxx, miny, maxy, minz, maxz]]

    dx = maxx - minx
    dy = maxy - miny
    dz = maxz - minz

    minx -= dx * offsetx
    maxx += dx * offsetx
    miny -= dy * offsety
    maxy += dy * offsety
    minz -= dz * offsetz
    maxz += dz * offsetz

    for ax in axs:
        ax.set_xlim(minz, maxz)
        ax.set_ylim(minx, maxx)
        ax.set_zlim(miny, maxy)
        ax.set_box_aspect((2, 1, 1))
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.view_init(elev=elv, azim=az)

    titles = ['before collision', 'during collision', 'after collision']

    for i in range(3):

        x1, y1, z1 = dat.get_particle(species='test1', var_list=['x', 'y', 'z'], iteration=iters[i])
        axs[i].scatter(z1, x1, y1, color=(0, 0, 1, 0.7), s=5)
        x2, y2, z2 = dat.get_particle(species='test2', var_list=['x', 'y', 'z'], iteration=iters[i])
        axs[i].scatter(z2, x2, y2, color=(1, 0, 0, 0.7), s=5)
        if sys.argv[2] == 'xy':
            z1.sort()
            midz = z1[len(z1)//2]
            poly = [np.array([(midz, sx, sy), (midz, sx, by), (midz, bx, by), (midz, bx, sy)])]
        elif sys.argv[2] == 'xz':
            y = sorted(list(y1) + list(y2))
            midy = y[len(y)//2]
            poly = [np.array([(sz, sx, midy), (bz, sx, midy), (bz, bx, midy), (sz, bx, midy)])]
        else:
            x = sorted(list(x1) + list(x2))
            midx = x[len(x)//2]
            poly = [np.array([(sz, midx, sy), (bz, midx, sy), (bz, midx, by), (sz, midx, by)])]
        polyobj = Poly3DCollection(poly, facecolors=(0, 0, 0, 0.3), edgecolors='black')
        axs[i].add_collection3d(polyobj)
        axs[i].set_title(titles[i])

    fig.suptitle(f'{sys.argv[2].upper()} slices throughout collision')
    legend_elements = [
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Beam 1 (electrons)'),
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Beam 2 (positrons)')
    ]
    fig.legend(handles=legend_elements, loc='upper left')

    plt.show()
