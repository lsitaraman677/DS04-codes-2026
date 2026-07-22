import sys
from openpmd_viewer import OpenPMDTimeSeries
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib

dat = OpenPMDTimeSeries(sys.argv[1])

elvchecks = [(20, -40)]

for elv, az in elvchecks:

    fig = plt.figure(figsize=(17, 5))
    axs = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]

    print(len(dat.iterations))
    iters = [0, 250, 500]
    iters = [dat.iterations[i] for i in iters]

    offsetx = 0.1
    offsety = 0.1
    offsetz = 0.1

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
        x2, y2, z2 = dat.get_particle(species='test2', var_list=['x', 'y', 'z'], iteration=iters[i])
        
        # 1. Combine coordinates into single arrays
        x_all = np.concatenate((x1, x2))
        y_all = np.concatenate((y1, y2))
        z_all = np.concatenate((z1, z2))
        
        # 2. Assign RGBA colors (Beam 1 = Blue, Beam 2 = Red)
        c1 = np.tile([0, 0, 1, 0.7], (len(x1), 1))
        c2 = np.tile([1, 0, 0, 0.7], (len(x2), 1))
        c_all = np.concatenate((c1, c2))
        
        # 3. Shuffle arrays so particles mix perfectly and don't paint red purely over blue
        shuffle_idx = np.random.permutation(len(x_all))
        x_all, y_all, z_all, c_all = x_all[shuffle_idx], y_all[shuffle_idx], z_all[shuffle_idx], c_all[shuffle_idx]

        if sys.argv[2] == 'xy':
            mid = np.median(z1)
            poly = [np.array([(mid, sx, sy), (mid, sx, by), (mid, bx, by), (mid, bx, sy)])]
            # Change '<' to '>=' if the wrong side gets darkened based on your camera angle!
            mask_back = z_all < mid
            mask_front = z_all >= mid
            
        elif sys.argv[2] == 'xz':
            mid = np.median(y_all)
            poly = [np.array([(sz, sx, mid), (bz, sx, mid), (bz, bx, mid), (sz, bx, mid)])]
            mask_back = y_all < mid
            mask_front = y_all >= mid
            
        else:
            mid = np.median(x_all)
            poly = [np.array([(sz, mid, sy), (bz, mid, sy), (bz, mid, by), (sz, mid, by)])]
            mask_back = x_all < mid
            mask_front = x_all >= mid

        # 4. Manually darken the colors of the particles behind the plane
        # Multiply the R, G, B channels (indices 0, 1, 2) by 0.3. Leave Alpha (index 3) alone.
        darken_factor = 0.65
        c_back = c_all[mask_back].copy()
        c_back[:, :3] *= darken_factor 
        
        c_front = c_all[mask_front]

        # 5. Plot the back particles, then the plane, then the front particles
        axs[i].scatter(z_all[mask_back], x_all[mask_back], y_all[mask_back], c=c_back, s=5)
        
        polyobj = Poly3DCollection(poly, facecolors=(0, 0, 0, 0.15), edgecolors='black')
        axs[i].add_collection3d(polyobj)
        
        axs[i].scatter(z_all[mask_front], x_all[mask_front], y_all[mask_front], c=c_front, s=5)
        
        axs[i].set_title(titles[i])

    fig.suptitle(f'{sys.argv[2].upper()} slices throughout collision')
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Beam 1 (electrons)'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Beam 2 (positrons)')
    ]
    fig.legend(handles=legend_elements, loc='upper left')

    plt.show()
