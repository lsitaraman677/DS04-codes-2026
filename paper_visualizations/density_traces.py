from matplotlib import pyplot as plt
import numpy as np
import sys
from openpmd_viewer import OpenPMDTimeSeries

path = sys.argv[1]
ts = OpenPMDTimeSeries(path)

width = 250
height = int(2.4 * width)

iters = ts.iterations
x, y, z = ts.get_particle(species = "beam1", var_list = ["x", 'y', "z"], iteration = 0)
x2, y2, z2 = ts.get_particle(species = "beam2", var_list = ["x", 'y', "z"], iteration = 0)
x = np.array(x)
y = np.array(y)
z = np.array(z)
x2 = np.array(x2)
y2 = np.array(y2)
z2 = np.array(z2)
mi = lambda a, b: min(np.min(a), np.min(b))
ma = lambda a, b: max(np.max(a), np.max(b))
minx, maxx = mi(x, x2), ma(x, x2)
miny, maxy = mi(y, y2), ma(y, y2)
minz, maxz = mi(z, z2), ma(z, z2)

species = ts.avail_species
print(species)

def get_ids(n, spec):
    res = ts.get_particle(species=spec, var_list=['id'], iteration=ts.iterations[-1])[0]
    return np.array(res)[np.linspace(0, len(res)-1, n, dtype=np.int32)]

n = 200

ids = [get_ids(n, i) for i in species]

vs = ['x', 'y', 'z', 'px', 'py', 'pz', 'ex', 'ey', 'ez', 'bx', 'by', 'bz', 'weight']

xz_density = np.empty(shape=(len(iters), height, width))
yz_density = np.empty(shape=(len(iters), height, width))
partvals = np.empty(shape=(len(iters), len(species), n, len(vs)))

idx = 0
for it in iters:
    for i in range(len(species)):
        spec = species[i]
        cur = ts.get_particle(species=spec, var_list=vs+['id'], iteration=it)
        cur, curid = cur[:-1], cur[-1]
        cur = np.array(cur).T
        mask = np.isin(curid, ids[i])
        partvals[idx, i] = cur[mask]
    xp, yp, zp = ts.get_particle(species="beam1", var_list = ["x", 'y', "z"], iteration = it)
    x2p, y2p, z2p = ts.get_particle(species="beam2", var_list = ["x", 'y', "z"], iteration = it)
    xp = np.array(xp)
    yp = np.array(yp)
    zp = np.array(zp)
    x2p = np.array(x2p)
    y2p = np.array(y2p)
    z2p = np.array(z2p)
    x = np.astype((xp-minx)/(maxx-minx+1e-10) * height, np.int32)
    y = np.astype((yp-miny)/(maxy-miny+1e-10) * height, np.int32)
    z = np.astype((zp-minz)/(maxz-minz+1e-10) * width, np.int32)
    x2 = np.astype((x2p-minx)/(maxx-minx+1e-10) * height, np.int32)
    y2 = np.astype((y2p-miny)/(maxy-miny+1e-10) * height, np.int32)
    z2 = np.astype((z2p-minz)/(maxz-minz+1e-10) * width, np.int32)
    i1y = y * width + z
    i2y = y2 * width + z2
    i1x = x * width + z
    i2x = x2 * width + z2
    i1y[i1 >= width * height] = 0
    i1y[i1 < 0] = 0
    i2y[i2 >= width * height] = 0
    i2y[i2 < 0] = 0
    i1x[i1 >= width * height] = 0
    i1x[i1 < 0] = 0
    i2x[i2 >= width * height] = 0
    i2x[i2 < 0] = 0
    bin1 = np.bincount(i1x, minlength = width * height)
    bin2 = np.bincount(i2x, minlength = width * height)
    densitiesx = np.reshape(bin2-bin1, shape = (height, width))
    bin1 = np.bincount(i1y, minlength = width * height)
    bin2 = np.bincount(i2y, minlength = width * height)
    densitiesy = np.reshape(bin2-bin1, shape = (height, width))

    xz_density[idx] = densitiesx
    yz_density[idx] = densitiesy

    print(f'finished iteration {it}')
    idx += 1


pre = sys.argv[2]

np.save(f'{pre}xz_densities.npy', xz_density)
np.save(f'{pre}yz_densities.npy', yz_density)
np.save(f'{pre}particle_data.npy', partvals)
np.save(f'{pre}otherthings.npy', np.array([species, vs, n], dtype=object))
