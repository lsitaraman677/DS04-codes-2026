from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import cv2
import sys
import random

class Particle:

    def __init__(self, pos, charge):
        self.charge = charge
        col = 'red' if (charge > 0) else 'blue'
        self.circle = Circle(pos, 0.03 * abs(charge), facecolor=col, edgecolor='black')
        self.line = Line2D([], [], color='black', linewidth=2*abs(charge))
        dist = 0.013 * abs(charge)
        self.pts = [(dist, 0), (0, dist), (-dist, 0), (0, -dist)] if (charge > 0) else [(dist, 0), (-dist, 0)]
        self.update_patches()

    def update_patches(self):
        x = []
        y = []
        xc, yc = self.pos()
        for i in range(2 * len(self.pts) + 1):
            if i%2 == 0:
                x.append(xc)
                y.append(yc)
            else:
                idx = i // 2
                x.append(self.pts[idx][0] + xc)
                y.append(self.pts[idx][1] + yc)
        self.line.set_data(x, y)

    def pos(self):
        return self.circle.center

    def goto(self, newpos):
        self.circle.center = newpos

    def get_patches(self):
        return [self.circle, self.line]

particles = [Particle((0.5, 0.5), 1)]

fig, ax = plt.subplots()
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.08, 1.08)
ax.set_aspect('equal')

bins = 14
x = np.linspace(0, 1, bins+1)
x, y = np.meshgrid(x, x)

def calc_uvc():
    u = np.zeros(shape=x.shape)
    v = np.zeros(shape=x.shape)
    for particle in particles:
        xc, yc = particle.pos()
        diffx = x - xc
        diffy = y - yc
        distsq = np.square(diffx) + np.square(diffy)
        mask = distsq < (0.03**2) * abs(particle.charge)
        distsq[mask] = 1e15
        fact = np.pow(distsq, -0.7) * particle.charge
        u += diffx * fact
        v += diffy * fact
    c = np.sqrt(np.square(u) + np.square(v))
    return u, v, c

quiv = ax.quiver(x, y, *calc_uvc(), cmap='viridis')
for particle in particles:
    for artist in particle.get_patches():
        if isinstance(artist, Line2D):
            ax.add_line(artist)
        else:
            ax.add_patch(artist)

try:
    vidname = sys.argv[1]
except IndexError:
    vidname = ''

plt.ion()
plt.show()

plt.pause(0.1)

start = False
def clicked(event):
    global start
    if event.button == 3:
        start = True

fig.canvas.mpl_connect('button_press_event', clicked)

while not start:
    plt.pause(0.01)

fps = 30
dt = 1 / fps

if vidname:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    height, width, _ = frame.shape
    video = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

T = 1.4
firstT = 1
lastT = 1

t = 0
goalpos = (random.random(), random.random())
count = 5
curcount = 0
b = np.pi / T
prevpos = particles[0].pos()
diffx = goalpos[0] - prevpos[0]
diffy = goalpos[1] - prevpos[1]
dist = (diffx**2 + diffy**2)**0.5
done = False

while True:
    if (t > firstT) and (not done):
        checkcount = (t - firstT) // T
        if checkcount > curcount:
            curcount += 1
            if curcount == count:
                done = True
                continue
            prevpos = goalpos
            goalpos = (random.random(), random.random()) if (curcount < count - 1) else (0.5, 0.5)
            diffx = goalpos[0] - prevpos[0]
            diffy = goalpos[1] - prevpos[1]
            dist = (diffx**2 + diffy**2)**0.5
        curt = (t - firstT) % T
        d = dist * 0.5 * (1 - np.cos(b * curt))
        fact = d / dist
        newpos = (prevpos[0] + diffx * fact, prevpos[1] + diffy * fact)
        particles[0].goto(newpos)
        particles[0].update_patches()
        quiv.set_UVC(*calc_uvc())
        fig.canvas.draw_idle()
        plt.pause(0.01)
    if done:
        if t > firstT + T * count + lastT:
            break
    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)
    t += dt

if vidname:
    video.release()
