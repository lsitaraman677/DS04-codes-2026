from matplotlib import pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import numpy as np
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
thread = comm.Get_rank()
num_threads = comm.Get_size() - 1

def trimshape(arr, newshape):
    oldshape = arr.shape
    idxs = []
    for i in range(len(oldshape)):
        even = np.linspace(0, oldshape[i]-1, newshape[i], dtype=np.int32)
        paddedshape = tuple([even.shape[0]] + [1 for _ in range(len(oldshape) - 1 - i)])
        idxs.append(even.reshape(paddedshape))
    return arr[*idxs]

if thread == 0:
    print('loading data')

vecs = OpenPMDTimeSeries(sys.argv[1])
pts = OpenPMDTimeSeries(sys.argv[2])
iters = np.arange(0, 501, 4, dtype=np.int32)

num_pts = 1000
num_strong = 50

quiver_shape = (10, 30)

if thread == 0:
    print(vecs.iterations)
    print(pts.iterations)

    print('loaded data')

if thread == 0:

    n_iters = len(iters)

    vals, griddata = vecs.get_field(field='E', coord='x', iteration=0)
    dimx, dimy, dimz = vals.shape
    #xgr, ygr, zgr = np.meshgrid(griddata.x, griddata.y, griddata.z, indexing='ij')

    print(vals.shape)

    print(len(griddata.x))
    print(len(griddata.y))
    print(len(griddata.z))

    zgr = trimshape(griddata.z, (quiver_shape[1],))
    xgr = trimshape(griddata.x, (quiver_shape[0],))

    zgr, xgr = np.meshgrid(zgr, xgr, indexing='xy')

    starty = griddata.y[0]
    endy = griddata.y[-1]
    leny = len(griddata.y)
    yidx = lambda y: int(((y - starty) / (endy - starty)) * leny)

    p = lambda s: sys.argv[3] + '/' + s + '.npy'
    np.save(p('zgr'), zgr)
    np.save(p('xgr'), xgr)
    np.save(p('iters'), iters)

    lastiter = round(iters[-1]/5) * 5
    firstiter = round(iters[0]/5) * 5
    lastids1 = np.array(pts.get_particle(species='beam1', var_list=['id'], iteration=lastiter)[0])
    firstids1 = np.array(pts.get_particle(species='beam1', var_list=['id'], iteration=firstiter)[0])
    lastids2 = np.array(pts.get_particle(species='beam2', var_list=['id'], iteration=lastiter)[0])
    firstids2 = np.array(pts.get_particle(species='beam2', var_list=['id'], iteration=firstiter)[0])
    candidates1 = lastids1[np.isin(lastids1, firstids1)]
    candidates2 = lastids2[np.isin(lastids2, firstids2)]
    print(f'{len(candidates1)} out of {len(lastids1)} were in beam1 from the start')
    print(f'{len(candidates2)} out of {len(lastids2)} were in beam1 from the start')
    ids1 = candidates1[np.linspace(0, len(candidates1)-1, num_pts//2, dtype=np.int32)]
    ids2 = candidates2[np.linspace(0, len(candidates2)-1, num_pts//2, dtype=np.int32)]

    subsets = np.empty(shape=(n_iters, num_pts, 3))
    ydiffs = np.empty(shape=(n_iters, num_pts))
    yranges = np.empty(shape=(n_iters,))
    cols = np.empty(shape=(n_iters, num_pts, 3))
    ues = np.empty(shape=(n_iters, *quiver_shape))
    ves = np.empty(shape=ues.shape)
    ubs = np.empty(shape=ues.shape)
    vbs = np.empty(shape=ues.shape)

    if num_threads > n_iters:
        itersperthread = np.zeros(shape=(num_threads,), dtype=np.int32)
        itersperthread[:n_iters] = 1
    else:
        itersperthread = np.full((num_threads,), (n_iters // num_threads))
        itersperthread[:n_iters%num_threads] += 1

    curstart = 0
    for i in range(num_threads):
        cursize = itersperthread[i]
        idxtup = (curstart, curstart + cursize)
        comm.send(idxtup, dest=i+1, tag=1)
        comm.send(ids1, dest=i+1, tag=101)
        comm.send(ids2, dest=i+1, tag=102)
        comm.send((starty, endy, leny), dest=i+1, tag=103)
        curstart += cursize

    curstart = 0
    for i in range(num_threads):
        cursize = itersperthread[i]
        curend = curstart + cursize
        comm.Recv(subsets[curstart:curend], source=i+1, tag=2)
        comm.Recv(ydiffs[curstart:curend], source=i+1, tag=3)
        comm.Recv(yranges[curstart:curend], source=i+1, tag=4)
        comm.Recv(cols[curstart:curend], source=i+1, tag=5)
        comm.Recv(ues[curstart:curend], source=i+1, tag=6)
        comm.Recv(ves[curstart:curend], source=i+1, tag=7)
        comm.Recv(ubs[curstart:curend], source=i+1, tag=8)
        comm.Recv(vbs[curstart:curend], source=i+1, tag=9)
        print(f'Received data for iterations {iters[curstart]} to {iters[curend-1]} from thread {i+1}')
        curstart = curend

    np.save(p('subsets'), subsets)
    np.save(p('ydiffs'), ydiffs)
    np.save(p('yranges'), yranges)
    np.save(p('cols'), cols)
    np.save(p('ues'), ues)
    np.save(p('ves'), ves)
    np.save(p('ubs'), ubs)
    np.save(p('vbs'), vbs)

    print(f'saved all values in folder {sys.argv[3]}')

else:

    def get_mask(idarr, idset):
        mask = np.isin(idarr, idset)
        return mask

    start, end = comm.recv(source=0, tag=1)
    n_iters = end - start

    if n_iters == 0:
        sys.exit()

    subsets = np.empty(shape=(n_iters, num_pts, 3))
    ydiffs = np.empty(shape=(n_iters, num_pts))
    yranges = np.empty(shape=(n_iters,))
    cols = np.empty(shape=(n_iters, num_pts, 3))
    ues = np.empty(shape=(n_iters, *quiver_shape))
    ves = np.empty(shape=ues.shape)
    ubs = np.empty(shape=ues.shape)
    vbs = np.empty(shape=ues.shape)

    ids1 = comm.recv(source=0, tag=101)
    ids2 = comm.recv(source=0, tag=102)
    starty, endy, leny = comm.recv(source=0, tag=103)
    yidx = lambda y: int(((y - starty) / (endy - starty)) * leny)

    for i in range(start, end):
        idx = i - start
        it = iters[i]
        vecsit = it
        ptsit = round(it/5) * 5
        curpts1 = np.array(pts.get_particle(species='beam1', var_list=['x', 'y', 'z'], iteration=ptsit))
        curpts2 = np.array(pts.get_particle(species='beam2', var_list=['x', 'y', 'z'], iteration=ptsit))
        curpts1 = np.append(curpts1, np.zeros(shape=(1, curpts1.shape[1])), axis=0).T
        curpts2 = np.append(curpts2, np.ones(shape=(1, curpts2.shape[1])), axis=0).T
        mask1 = get_mask(pts.get_particle(species='beam1', var_list=['id'], iteration=ptsit)[0], ids1)
        mask2 = get_mask(pts.get_particle(species='beam2', var_list=['id'], iteration=ptsit)[0], ids2)
        curpts1 = curpts1[mask1]
        curpts2 = curpts2[mask2]
        curpts = np.append(curpts1, curpts2, axis=0)
        idxs = np.argsort(curpts[:, 1])
        curpts = curpts[idxs]
        mididx = curpts.shape[0]//2
        midy = curpts[mididx, 1]
        yrange = (curpts[mididx + num_strong//2, 1] - curpts[mididx - num_strong//2, 1]) / 2
        ydiff = curpts[:, 1] - midy
        subset = curpts[:, :3]
        col = np.zeros(shape=(num_pts, 3))
        col[:, 0] = curpts[:, 3]
        col[:, 2] = 1 - curpts[:, 3]
        yi = yidx(midy)
        ue = trimshape(vecs.get_field(field='E', coord='z', iteration=vecsit)[0][:, yi, :].T, quiver_shape)
        ve = trimshape(vecs.get_field(field='E', coord='x', iteration=vecsit)[0][:, yi, :].T, quiver_shape)
        ub = trimshape(vecs.get_field(field='B', coord='z', iteration=vecsit)[0][:, yi, :].T, quiver_shape)
        vb = trimshape(vecs.get_field(field='B', coord='x', iteration=vecsit)[0][:, yi, :].T, quiver_shape)
        subsets[idx] = subset
        ydiffs[idx] = ydiff
        yranges[idx] = yrange
        cols[idx] = col
        ues[idx] = ue
        ves[idx] = ve
        ubs[idx] = ub
        vbs[idx] = vb

    comm.Send(subsets, dest=0, tag=2)
    comm.Send(ydiffs, dest=0, tag=3)
    comm.Send(yranges, dest=0, tag=4)
    comm.Send(cols, dest=0, tag=5)
    comm.Send(ues, dest=0, tag=6)
    comm.Send(ves, dest=0, tag=7)
    comm.Send(ubs, dest=0, tag=8)
    comm.Send(vbs, dest=0, tag=9)

