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

def loopfrompoints_fixed(pts, range_percent, max_dist):
    center = np.mean(pts, axis=0)
    pts2 = pts - center
    
    set1_pts = []
    set2_pts = []
    
    # Process each angle individually since array lengths will now vary
    for i in range(vecs.shape):
        v = vecs[:, i]
        pv = perpvecs[:, i]
        
        # Projections
        dots = np.dot(pts2, v)
        dists = np.square(np.dot(pts2, pv))
        
        # FILTER BY DISTANCE, NOT PERCENTAGE
        mask = dists < (max_dist ** 2)
        valid_dots = dots[mask]
        
        # Failsafe if the ray shoots completely into empty space
        if len(valid_dots) == 0:
            set1_pts.append(v * 0)
            set2_pts.append(v * 0)
            continue
            
        valid_dots.sort()
        
        # Calculate percentiles based on however many points we actually caught
        mid = len(valid_dots) // 2
        off = int((len(valid_dots) * range_percent * 0.01) / 2)
        
        l = max(0, mid - off)
        r = min(len(valid_dots) - 1, mid + off)
        
        set1_pts.append(v * valid_dots[r])
        set2_pts.append(v * valid_dots[l])
        
    set1 = np.array(set1_pts).T
    set2 = np.array(set2_pts).T
    return np.append(set1, set2, axis=1).T + center

# Test the fixed algorithm
pts = np.random.random((1000, 2))
pts[:, 1] *= 0.01

# max_dist is in absolute units now (e.g., 0.05 is a fixed radius)
loop = loopfrompoints_fixed(pts, range_percent=75, max_dist=0.05)

plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.plot(loop[:, 0], loop[:, 1], color='red')
plt.axis('equal') # Added to ensure the thin shape isn't visually distorted
plt.show()
