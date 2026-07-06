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
def loopfrompoints(pts, range_percent, close_percent, conv_percent=1):
    center = np.mean(pts, axis=0)
    pts2 = pts - center
    stdevx = np.std(pts2[:, 0])
    stdevy = np.std(pts2[:, 1])
    pts2[:, 0] /= stdevx
    pts2[:, 1] /= stdevy
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
    set2 = np.append(set2, set1[:, [0]], axis=1)
    loop = np.append(set1, set2, axis=1).T
    smoothloop = np.empty(shape=loop.shape)
    convnum = int(conv_percent * 0.01 * loop.shape[0])
    if convnum > 0:    
        for i in range(len(loop)):
            xsum = 0
            ysum = 0
            for j in range(convnum):
                xsum += loop[i-j, 0]
                ysum += loop[i-j, 1]
            xsum /= convnum
            ysum /= convnum
            smoothloop[i - (convnum//2), 0] = xsum
            smoothloop[i - (convnum//2), 1] = ysum
    else:
        smoothloop = loop
    smoothloop[:, 0] *= stdevx
    smoothloop[:, 1] *= stdevy
    smoothloop = np.append(smoothloop, smoothloop[0][np.newaxis, :], axis=0)
    return smoothloop + center

def fourier_blob(n=1500, center=(0, 0), base_r=1.0, n_harmonics=6, harmonic_decay=0.5, edge_noise=0.06, fill=True, seed=None):
    """A blobby/amoeba shape: radius(theta) = base_r * (1 + sum of random
    sine harmonics), points scattered near the boundary (or filled inside)."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(scale=harmonic_decay ** np.arange(1, n_harmonics + 1))
    phases = rng.uniform(0, 2 * np.pi, n_harmonics)

    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.ones(n)
    for k in range(n_harmonics):
        r += coeffs[k] * np.sin((k + 1) * theta + phases[k])
    r = base_r * np.clip(r, 0.15, None)

    if fill:
        # fill interior with density falling off toward the edge, like a
        # noisy blob rather than a hairline outline
        r = r * np.sqrt(rng.uniform(0, 1, n))
    r += rng.normal(scale=edge_noise * base_r, size=n)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])

pts = fourier_blob()
loop = loopfrompoints(pts, 100, 5, conv_percent=10)

plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.plot(loop[:, 0], loop[:, 1])
plt.show()
