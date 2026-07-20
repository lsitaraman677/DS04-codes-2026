from matplotlib import pyplot as plt
import numpy as np
import sys
from openpmd_viewer import OpenPMDTimeSeries
import cv2

path = sys.argv[1]
ts = OpenPMDTimeSeries(path)
iters = ts.iterations

def get_filtered_particles(ts, species, it, downsample_factor=10):
    try:
        data = ts.get_particle(species=species, var_list=["y", "z", "id"], iteration=it)
        if data is None:
            return np.array([]), np.array([])
            
        y, z, p_id = data
        y = np.array(y)
        z = np.array(z)
        p_id = np.array(p_id)
        
        mask = (p_id % downsample_factor == 0)
        return y[mask], z[mask]
    except Exception:
        return np.array([]), np.array([])

y, z = get_filtered_particles(ts, "beam1", 400, 1000)
y2, z2 = get_filtered_particles(ts, "beam2", 400, 1000)
y3, z3 = get_filtered_particles(ts, "pho1", 400, 5000)
y4, z4 = get_filtered_particles(ts, "pho2", 400, 5000)

y3 = np.append(y3, y4)
z3 = np.append(z3, z4)

def safe_min(*arrays):
    valid = [np.min(a) for a in arrays if len(a) > 0]
    return min(valid) if valid else -1.0

def safe_max(*arrays):
    valid = [np.max(a) for a in arrays if len(a) > 0]
    return max(valid) if valid else 1.0

miny = safe_min(y, y2, y3)
maxy = safe_max(y, y2, y3)
minz = safe_min(z, z2, z3)
maxz = safe_max(z, z2, z3)

diffz = maxz - minz
diffy = maxy - miny
offsetz = 0.5
offsety = 0.5
miny -= offsety * diffy
maxy += offsety * diffy
minz -= offsetz * diffz
maxz += offsetz * diffz

im = None
fig, ax = plt.subplots()
ax.set_xlim(minz, maxz)
ax.set_ylim(miny, maxy)
ax.set_xlabel('z')
ax.set_ylabel('y')
ax.set_title('Z vs Y, positrons, electrons, and emitted photons')

plt.ion()
plt.show()
plt.pause(0.1)

vidname = ''
try:
    vidname = sys.argv[2]
except IndexError:
    pass

start = False

def clicked(event):
    global start
    if event.button == 3:
        start = True

fig.canvas.mpl_connect('button_press_event', clicked)

while not start:
    plt.pause(0.01)

scat1 = ax.scatter([], [], color='blue', s=1)
scat2 = ax.scatter([], [], color='red',  s=1)
scat3 = ax.scatter([], [], color='green', s=1)

for it in iters:
    y, z = get_filtered_particles(ts, "beam1", it, DOWNSAMPLE_FACTOR)
    y2, z2 = get_filtered_particles(ts, "beam2", it, DOWNSAMPLE_FACTOR)
    
    y3, z3 = get_filtered_particles(ts, "pho1", it, DOWNSAMPLE_FACTOR)
    y4, z4 = get_filtered_particles(ts, "pho2", it, DOWNSAMPLE_FACTOR)
    
    y3 = np.append(y3, y4)
    z3 = np.append(z3, z4)

    scat1.set_offsets(np.column_stack((z, y)))
    scat2.set_offsets(np.column_stack((z2, y2)))
    scat3.set_offsets(np.column_stack((z3, y3)))

    fig.canvas.draw_idle()
    print(f'finished iteration {it}')
    plt.pause(0.01)

    if vidname:
        if im is None:
            frame = np.asarray(fig.canvas.buffer_rgba())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            h, w, _ = frame.shape
            video = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
            im = True

        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)

if vidname and im is not None:
    video.release()
