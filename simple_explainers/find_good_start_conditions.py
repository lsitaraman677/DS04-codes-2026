import numpy as np

charges = [1, 1, -1, -1]
accfact = 0.02
def countiters(startposes, startvels, dt):
    particles = np.copy(startposes)
    vels = np.copy(startvels)
    iters = 0
    minx = particles[:, 0].min()
    maxx = particles[:, 0].max()
    miny = particles[:, 1].min()
    maxy = particles[:, 1].max()
    while (maxx - minx < 1) and (maxy - miny < 1):
        for i1 in range(len(particles)-1):
            for i2 in range(i1+1, len(particles)):
                diff = particles[i2] - particles[i1]
                distsq = diff[0]**2 + diff[1]**2
                force = charges[i1] * charges[i2] / distsq
                acc = accfact * force * diff / (distsq**0.5)
                vels[i2] += acc * dt
                vels[i1] -= acc * dt
        particles += vels * dt
        minx = min(minx, particles[:, 0].min())
        maxx = max(maxx, particles[:, 0].max())
        miny = min(miny, particles[:, 1].min())
        maxy = max(maxy, particles[:, 1].max())
        iters += 1
    startposes[:, 0] += 0.5 - (maxx + minx) * 0.5
    startposes[:, 1] += 0.5 - (maxy + miny) * 0.5
    return iters

def disp(arr):
    pylist = [(float(i[0]), float(i[1])) for i in arr]
    return str(pylist)

import sys
best = 0
dt = (1/30)/100
#v = np.zeros((4,2))
while True:
    #x = np.array([(0.9499035240578857, 0.9616119559206568), (0.6380616921336619, 0.5984737362825525), (0.20086554125669065, 0.34449539431388176), (0.5763804661021011, 0.746975304157472)])
    x = np.random.random((4, 2))
    v = np.random.random((4, 2)) * 0.05
    try:
        iters = countiters(x, v, dt)
    except KeyboardInterrupt:
        sys.exit()
    if iters > best:
        print(f'new best! {iters} iters!\nx: {disp(x)}\nv: {disp(v)}')
        best = iters
    iters += 1
