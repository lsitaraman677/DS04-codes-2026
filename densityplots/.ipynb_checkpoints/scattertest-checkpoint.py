from openpmd_viewer import OpenPMDTimeSeries 
import sys
from matplotlib import pyplot as plt

ts = OpenPMDTimeSeries(sys.argv[1])

fig, ax = plt.subplots()

scat = None

plt.ion()
plt.show()

for it in ts.iterations:

