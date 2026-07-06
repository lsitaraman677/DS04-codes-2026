import numpy as np
import matplotlib.pyplot as plt

angs = np.linspace(0, np.pi, 200)
vecs = []
perpvecs = []
for i in angs:
    vecs.append([np.cos(i), np.sin(i)])
    perpvecs.append([-np.sin(i), np.cos(i)])
vecs = np.array(vecs).T
perpvecs = np.array(perpvecs).T

def smooth_circular(arr, window):
    """Circular moving-average smoothing along axis 0. arr shape (N, D)."""
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]], axis=0)
    kernel = np.ones(window) / window
    out = np.zeros_like(arr)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode='valid')[:arr.shape[0]]
    return out

def loopfrompoints(pts, range_percent, close_percent, smooth_window=5):
    # --- whiten so anisotropic scale doesn't distort "closeness" ---
    center = np.mean(pts, axis=0)
    scale = np.std(pts, axis=0)
    scale[scale == 0] = 1.0
    pts_w = (pts - center) / scale  # whitened, isotropic-ish

    dots = np.matmul(pts_w, vecs)
    dists = np.square(np.matmul(pts_w, perpvecs))
    idxs = np.argsort(dists, axis=0)
    dots = dots[idxs, np.arange(dots.shape[1])][:int(close_percent * 0.01 * dots.shape[0])]
    dots.sort(axis=0)
    mid = dots.shape[0] // 2
    off = int((dots.shape[0] * range_percent * 0.01) / 2)
    l = mid - off
    if l < 0:
        l = 0
    r = mid + off
    if r >= dots.shape[0]:
        r = dots.shape[0] - 1

    set1 = (vecs * dots[r]).T  # whitened-space outer boundary
    set2 = (vecs * dots[l]).T  # whitened-space inner/other boundary

    loop_w = np.append(set1, set2, axis=0)
    loop = loop_w * scale + center  # un-whiten back to original coordinates

    loop = smooth_circular(loop, smooth_window)
    return loop

np.random.seed(0)
pts = np.random.random((1000, 2))
pts[:, 1] *= 0.01

loop = loopfrompoints(pts, 100, 5, smooth_window=50)

plt.figure(figsize=(12,5))
plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.plot(np.append(loop[:,0], loop[0,0]), np.append(loop[:,1], loop[0,1]))

plt.show()
print("done")
