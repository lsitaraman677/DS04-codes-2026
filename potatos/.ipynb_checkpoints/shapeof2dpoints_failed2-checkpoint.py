from matplotlib import pyplot as plt
import numpy as np

angs = np.linspace(0, np.pi, 200)
vecs = []
perpvecs = []
for i in angs:
    vecs.append([np.cos(i), np.sin(i)])
    perpvecs.append([-np.sin(i), np.cos(i)])
vecs = np.array(vecs).T
perpvecs = np.array(perpvecs).T
def loopfrompoints(pts, range_percent, close_percent):
    center = np.mean(pts, axis=0)
    pts2 = pts - center
    dots = np.matmul(pts2, vecs)
    dists = np.square(np.matmul(pts2, perpvecs))
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
    set1 = vecs * dots[r]
    set2 = vecs * dots[l]
    return np.append(set1, set2, axis=1).T + center


pts = np.random.random((1000, 2))
pts[:, 1] *= 0.01
loop = loopfrompoints(pts, 90, 5)

plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.plot(loop[:, 0], loop[:, 1])
plt.show()
