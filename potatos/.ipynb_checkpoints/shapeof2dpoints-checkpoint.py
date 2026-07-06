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
    avgdist = np.sqrt(np.sum(np.square(pts2)) / pts2.shape[0])
    dots = np.matmul(pts2, vecs)
    dists = np.square(np.matmul(pts2, perpvecs))
    vecmins = []
    vecmas = []
    for col in range(dots.shape[1]):
        closedots = []
        for row in range(dots.shape[0]):
            if abs(dists[row, col]) < (avgdist * close_percent * 0.01):
                closedots.append(dots[row, col])
        mid = len(closedots) // 2
        off = int((len(closedots) * range_percent * 0.01) / 2)
        r = mid + off
        if r >= len(closedots):
            r = len(closedots) - 1
        l = mid - off
        if l < 0:
            l = 0
        closedots.sort()
        vecmins.append(closedots[l])
        vecmas.append(closedots[r])
    vecmins = np.array(vecmins)
    vecmas = np.array(vecmas)
    set1 = vecs * vecmas
    set2 = vecs * vecmins
    return np.append(set1, set2, axis=1).T + center


pts = np.random.random((1000, 2))
pts[:, 1] *= 0.01
loop = loopfrompoints(pts, 1000, 0.01)

plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.plot(loop[:, 0], loop[:, 1])
plt.show()
