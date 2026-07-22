from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
numthreads = comm.Get_size()
rank = comm.Get_rank()

def trimshape(arr, newshape):
    oldshape = arr.shape
    idxs = []
    for i in range(len(oldshape)):
        even = np.linspace(0, oldshape[i]-1, newshape[i], dtype=np.int32)
        paddedshape = tuple([even.shape[0]] + [1 for _ in range(len(oldshape) - 1 - i)])
        idxs.append(even.reshape(paddedshape))
    return arr[*idxs]

if num_threads == 1:

    inputfile = sys.argv[1]

    with open(inputfile, 'r') as f:
        for i in f.readlines():
            exec(i)

    fields = OpenPMDTimeSeries(field_data)
    particles = OpenPMDTimeSeries(particle_data)

    fielditers = fields.iterations
    particleiters = particles.iterations
    
    efield = np.empty(shape=(iters, *field_size, 3))
    bfield = np.empty(shape=(iters, *field_size, 3))
    j = 0
    idx = 0
    saves = {}
    for i in np.linspace(0, particlesiters[-1], iters):
        while fielditers[j] <= i:
            j += 1
        if (j-1) in saves:
            prevdat_e, prevdat_b = saves[j-1]
        else:
            prevdat_e = np.empty(shape=(*orig_field_size, 3))
            prevdat_e[:, :, :, 2] = fields.get_field(field='E', coord='x', iteration=fielditers[j-1])[0]
            prevdat_e[:, :, :, 1] = fields.get_field(field='E', coord='y', iteration=fielditers[j-1])[0]
            prevdat_e[:, :, :, 0] = fields.get_field(field='E', coord='z', iteration=fielditers[j-1])[0]
            prevdat_e = trimshape(prevdat_e, (*orig_field_size, 3)) 
            prevdat_b = np.epmty(shape=(*orig_field_size, 3))
            prevdat_b[:, :, :, 2] = fields.get_field(field='B', coord='x', iteration=fielditers[j-1])[0]
            prevdat_b[:, :, :, 1] = fields.get_field(field='B', coord='y', iteration=fielditers[j-1])[0]
            prevdat_b[:, :, :, 0] = fields.get_field(field='B', coord='z', iteration=fielditers[j-1])[0]
            prevdat_b = trimshape(prevdat_b, (*orig_field_size, 3))
            saves[j-1] = prevdat_e, prevdat_b
        if j in saves:
            nextdat_e, nextdat_b = saves[j]
        else:
            nextdat_e = np.empty(shape=(*orig_field_size, 3))
            nextdat_e[:, :, :, 2] = fields.get_field(field='E', coord='x', iteration=fielditers[j])[0]
            nextdat_e[:, :, :, 1] = fields.get_field(field='E', coord='y', iteration=fielditers[j])[0]
            nextdat_e[:, :, :, 0] = fields.get_field(field='E', coord='z', iteration=fielditers[j])[0]
            nextdat_e = trimshape(prevdat_e, (*orig_field_size, 3))
            nextdat_b = np.epmty(shape=(*orig_field_size, 3))
            nextdat_b[:, :, :, 2] = fields.get_field(field='B', coord='x', iteration=fielditers[j])[0]
            nextdat_b[:, :, :, 1] = fields.get_field(field='B', coord='y', iteration=fielditers[j])[0]
            nextdat_b[:, :, :, 0] = fields.get_field(field='B', coord='z', iteration=fielditers[j])[0]
            nextdat_b = trimshape(prevdat_b, (*orig_field_size, 3))
            saves[j] = nextdat_e, prevdat_b
        t = (i - fielditers[j-1]) / (fielditers[j] - fielditers[j-1])
        curfield_e = prevdat_e * t + nextdat_e * (1 - t)  
        curfield_b = prevdat_b * t + nextdat_b * (1 - t)
        efield[idx] = curfield_e
        bfield[idx] = curfield_b
        idx += 1

     

        


        





