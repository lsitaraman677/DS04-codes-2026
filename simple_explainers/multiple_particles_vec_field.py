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
        return np.array(self.circle.center)

    def goto(self, newpos):
        self.circle.center = newpos
        self.update_patches()

    def get_patches(self):
        return [self.circle, self.line]

#poses = [(0.11719296657906453, 0.6368309048043945), (0.7294859005587234, 0.6098789469613767), (0.6259181322999711, 0.8323195269651893), (0.9263883415706254, 0.14040871188284731)]
#poses = [(0.019166654138441874, 0.738375777910376), (0.8582146171372398, 0.4438222143640329), (0.359556082542422, 0.5437405965827878), (0.798960004871276, 0.18590536689240345)]
poses = [(0.8343422873749883, 0.8377035014521784), (0.5225004554507645, 0.4745652818140741), (0.08530430457379323, 0.22058693984540334), (0.4608192294192037, 0.6230668496889936)]
#poses = [(0.13598078118846257, -0.13207474117997098), (-0.1758610507357612, -0.4952129608180753), (-0.6130572016127325, -0.7491913027867461), (-0.23754227676732198, -0.3467113929431558)]
#vels = [(0.012659120751590104, 0.09409569210386864), (0.04181997385860114, 0.04430463173851389), (0.09057348057859466, 0.06428107346249377), (0.040516130747464574, 0.02096195378591197)]
#poses = [(0.8221832073822439, 0.06440239983072116), (0.2945556431012767, 0.13481671444924498), (0.7571898825728925, 0.36186510724452026), (0.049904871880074864, 0.6166320326812273)]
#vels = [(0.0476570787074787, 0.009102532704553424), (-0.020381534407411778, 0.011337141276636565), (-0.03248831995595771, -0.009817904809980328), (0.036486617260769676, 0.04652007898842789)]
particles = [Particle(poses[i], 1 if (i < 2) else -1) for i in range(4)]

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

vels = np.array([[1, 0.85], [-0.9, -1], [0, 0.2], [2.07, -1.02]], dtype=np.float64) * 0
#vels = np.array(vels)
accfact = 0.02
times = 100

iters = 0

while True:

    for _ in range(times):

        for i1 in range(len(particles)-1):
            for i2 in range(i1+1, len(particles)):
                diff = particles[i2].pos() - particles[i1].pos()
                distsq = diff[0]**2 + diff[1]**2
                force = particles[i1].charge * particles[i2].charge / distsq
                acc = accfact * force * diff / (distsq**0.5)
                vels[i2] += acc * dt/times
                vels[i1] -= acc * dt/times

        for i in range(len(particles)):
            particles[i].goto(particles[i].pos() + vels[i] * dt/times)

        iters += 1

    if iters > 32500:
        break

    quiv.set_UVC(*calc_uvc())

    fig.canvas.draw_idle()
    plt.pause(0.01)

    if vidname:
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame)

if vidname:
    video.release()
