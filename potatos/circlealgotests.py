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

import numpy as np
import scipy.interpolate as interp
from scipy.stats import gaussian_kde
from skimage import measure

def get_smooth_density_loop(points, percentage=100, grid_res=150, smooth_factor=0.05):
    """
    Generates a smooth, closed concave loop around a point cloud based on a percentage threshold.

    Parameters:
    - points: array-like of shape (N, 2)
    - percentage: float (0 to 100). 100 captures the outer boundary; lower numbers contract inward.
    - grid_res: resolution of the internal density grid (higher = more detailed)
    - smooth_factor: smoothing intensity for the final spline
    """
    points = np.asarray(points)
    x, y = points[:, 0], points[:, 1]

    # 1. Compute Kernel Density Estimation (KDE) to map the amoeba shape
    kde = gaussian_kde(points.T)

    # Create grid padded slightly past the min/max coordinates
    x_margin = (x.max() - x.min()) * 0.15
    y_margin = (y.max() - y.min()) * 0.15
    xi, yi = np.linspace(x.min() - x_margin, x.max() + x_margin, grid_res), \
             np.linspace(y.min() - y_margin, y.max() + y_margin, grid_res)
    X, Y = np.meshgrid(xi, yi)

    # Evaluate density across the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # 2. Map the user's percentage to a density threshold
    # Evaluate density exactly at the given point coordinates to find the data's range
    point_densities = kde(points.T)

    if percentage >= 100:
        # Lowest density near any real point (outermost edge)
        threshold = np.min(point_densities) * 0.9
    elif percentage <= 0:
        # Highest density core
        threshold = np.max(point_densities)
    else:
        # Linearly map percentage to the distribution of point densities
        threshold = np.percentile(point_densities, 100 - percentage)

    # 3. Extract the concave boundary contour at that threshold
    contours = measure.find_contours(Z, threshold)
    if not contours:
        raise ValueError("Could not find a valid loop. Try increasing the percentage.")

    # Pick the largest contour (main amoeba body)
    contour = max(contours, key=len)

    # Convert grid index coordinates back to data coordinates
    contour_x = xi[0] + (contour[:, 1] / (grid_res - 1)) * (xi[-1] - xi[0])
    contour_y = yi[0] + (contour[:, 0] / (grid_res - 1)) * (yi[-1] - yi[0])

    # Ensure the loop is closed for interpolation
    if not np.allclose(contour_x[0], contour_x[-1]) or not np.allclose(contour_y[0], contour_y[-1]):
        contour_x = np.append(contour_x, contour_x[0])
        contour_y = np.append(contour_y, contour_y[0])

    # 4. Smooth the jagged pixel contour into a continuous loop using B-splines
    tck, u = interp.splprep([contour_x, contour_y], s=smooth_factor, per=True)
    u_new = np.linspace(0, 1, 300) # 300 points for a highly smooth render
    smooth_x, smooth_y = interp.splev(u_new, tck)

    return np.column_stack((smooth_x, smooth_y))

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

def multi_lobed_blob(n=1500, center=(0, 0), base_r=1.0, n_lobes=4,
                      lobe_strength=0.5, edge_noise=0.05, seed=None):
    """Star/clover-like amoeba with distinct lobes (deeper concavities than
    pure random Fourier noise)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = base_r * (1 + lobe_strength * np.cos(n_lobes * theta + rng.uniform(0, 0.3)))
    r *= np.sqrt(rng.uniform(0, 1, n))
    r += rng.normal(scale=edge_noise * base_r, size=n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])

import random
pts = fourier_blob(n_harmonics=7)
pts[:, 1] *= 0.01
loop = loopfrompoints(pts, 100, 5, conv_percent=10)
loop2 = get_smooth_density_loop(pts, 93)
plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.plot(loop[:, 0], loop[:, 1])
plt.plot(loop2[:, 0], loop2[:, 1])
plt.show()
