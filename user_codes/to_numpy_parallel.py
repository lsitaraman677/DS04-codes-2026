from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
thread = comm.Get_rank()
num_threads = comm.Get_size() 

ts = OpenPMDTimeSeries(sys.argv[1])

particles = []
i = 3
while sys.argv[i] != ',':
    particles.append(sys.argv[i])
    i += 1
var_list = []
for i in sys.argv[i+1:]:
    var_list.append(i)

print(particles)
print(var_list)

if thread == 0:

    print(f'{num_threads} thread{'s' if num_threads > 1 else ''} active.')

    name = sys.argv[2]
    
    if num_threads >= len(ts.iterations):
        vals = np.ones(num_threads, dtype=np.int32)
        vals[len(ts.iterations):] = 0
    else:
        amount = len(ts.iterations) // num_threads
        remainder = len(ts.iterations) % num_threads
        vals = np.ones(num_threads, dtype=np.int32) * amount
        vals[:remainder] += 1

    cur = vals[0]
    for i in range(1, num_threads):
        comm.send((cur, cur + vals[i]), dest=i, tag=1)
        cur += vals[i]

    cur = vals[0]
    ma = 0
    for i in range(1, num_threads):
        if vals[i] == 0:
            continue
        num = comm.recv(res[cur:cur+vals[i]], source=i, tag=101)
        if num > ma:
            ma = num
        cur += vals[i]

    for i in range(vals[0]):
        maxval = 0
        for p in range(len(particles)):
            cur = len(ts.get_particle(species=particles[p], var_list=[var_list[0]], iteration=ts.iterations[i])[0])
            if cur > maxval:
                maxval = cur
    if maxval > ma:
        ma = maxval

    print(f'found max number of particles: {ma}')

    res = np.empty(shape=(len(ts.iterations), len(particles), len(var_list), maxval))
    lens = np.empty(shape=(len(ts.iterations), len(particles)))

    for i in range(1, num_threads):
        if vals[i] == 0:
            continue
        comm.send(ma, dest=i, tag=2)

    cur = vals[0]
    for i in range(1, num_threads):
        if vals[i] == 0:
            continue
        comm.Recv(res[cur:cur+vals[i]], source=i, tag=102) 
        com.Recv(lens[cur:cur+vals[i]], source=i, tag=103)
        cur += vals[i]

    for i in range(vals[0]):
        for p in range(len(particles)):
            cur = np.array(ts.get_particle(species=particles[p], var_list=var_list, iteration=ts.iterations[i]))
            l = cur.shape[1]
            lens[i, p] = l
            res[i, p, :, :l] = cur

    np.save(f'{name}_lengths.npy', lens)
    np.save(f'{name}_{'_'.join(var_list)}.npy', res)

    print(f'Done. Data file is: {f'{name}_{'_'.join(var_list)}.npy'}. Lengths file is: {f'{name}_{'_'.join(var_list)}.npy'}')

else:

    its = comm.recv(source=0, tag=1)

    for i in range(its[0], its[1]):
        maxval = 0
        for p in range(len(particles)):
            cur = len(ts.get_particle(species=particles[p], var_list=[var_list[0]], iteration=ts.iterations[i])[0])
            if cur > maxval:
                maxval = cur
    if its[0] == its[1]:
        sys.exit()
    comm.send(maxval, dest=0, tag=101)

    ma = comm.recv(source=0, tag=2)
    data = np.empty(shape=(its[1] - its[0], len(particles), len(var_list), ma))
    lendat = np.empty(shape=(its[1] - its[0], len(particles)))
    for i in range(vals[0]):
        for p in range(len(particles)):
            cur = np.array(ts.get_particle(species=particles[p], var_list=var_list, iteration=ts.iterations[i]))
            l = cur.shape[1]
            lendat[i, p] = l
            data[i, p, :, :l] = cur
    comm.Send(data, dest=0, tag=102)
    comm.Send(lendat, dest=0, tag=103)





