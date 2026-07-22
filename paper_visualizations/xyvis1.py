from matplotlib import pyplot as plt
import numpy as np
import sys
import matplotlib

def copyq(original_quiver, new_ax):
    """
    Copies a matplotlib Quiver object exactly onto a new Axes.

    Parameters:
    original_quiver (matplotlib.quiver.Quiver): The quiver object to copy.
    new_ax (matplotlib.axes.Axes): The target axes to draw the new quiver on.

    Returns:
    matplotlib.quiver.Quiver: The newly created quiver object.
    """
    # 1. Extract coordinate and vector data
    X = original_quiver.X
    Y = original_quiver.Y
    U = original_quiver.U
    V = original_quiver.V

    # 2. Extract mapped color data (if a colormap was used)
    C = original_quiver.get_array()

    # 3. Extract all geometric and styling parameters
    kwargs = {
        'units': getattr(original_quiver, 'units', 'width'),
        'angles': getattr(original_quiver, 'angles', 'uv'),
        'scale': getattr(original_quiver, 'scale', None),
        'scale_units': getattr(original_quiver, 'scale_units', None),
        'width': getattr(original_quiver, 'width', None),
        'headwidth': getattr(original_quiver, 'headwidth', 3),
        'headlength': getattr(original_quiver, 'headlength', 5),
        'headaxislength': getattr(original_quiver, 'headaxislength', 4.5),
        'minshaft': getattr(original_quiver, 'minshaft', 1),
        'minlength': getattr(original_quiver, 'minlength', 1),
        'pivot': getattr(original_quiver, 'pivot', 'tail'),
        'zorder': original_quiver.get_zorder(),
        'alpha': original_quiver.get_alpha(),
    }

    # Filter out None values so Matplotlib can use its native defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # 4. Handle colors
    if C is None:
        # If no array was passed, grab the static face colors (RGBA array)
        colors = original_quiver.get_facecolor()
        if len(colors) > 0:
            kwargs['color'] = colors
    else:
        # If an array was passed, carry over the colormap and normalization
        kwargs['cmap'] = original_quiver.get_cmap()
        kwargs['norm'] = original_quiver.norm

    # 5. Draw the new quiver on the target axes
    if C is not None:
        new_quiver = new_ax.quiver(X, Y, U, V, C, **kwargs)
    else:
        new_quiver = new_ax.quiver(X, Y, U, V, **kwargs)

    return new_quiver

print('loading saved data')

load = lambda s: np.load(sys.argv[1] + '/' + s + '.npy')

xgr = load('xgr')
ygr = load('ygr')
subsets = load('subsets')
zdiffs = load('zdiffs')
zranges = load('zranges')
cols = load('cols')
ubs = load('ubs')
vbs = load('vbs')
ues = load('ues')
ves = load('ves')
iters = load('iters')

xdiff = xgr[-1, -1] - xgr[0, 0]
xmin, xmax = xgr[0, 0] - 0.05 * xdiff, xgr[-1, -1] + 0.05 * xdiff
ydiff = ygr[-1, -1] - ygr[0, 0]
ymin, ymax = ygr[0, 0] - 0.05 * ydiff, ygr[-1, -1] + 0.05 * ydiff

quiver_shape = xgr.shape

print('loaded saved data')

fig, axs = plt.subplots(1, 3)

for ax in axs:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    #ax.set_aspect('equal')
    ax.set_box_aspect(quiver_shape[0] / quiver_shape[1])

legend_elements = [
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Beam 1 (electrons)'),
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Beam 2 (positrons)')
]
fig.legend(handles=legend_elements)
fig.suptitle('Magnetic field at an XY slice taken at the median Z position of the electron beam')

k = 0.05
i = 0
emags = np.sqrt(np.square(ues) + np.square(ves))
bmags = np.sqrt(np.square(ubs) + np.square(vbs))
extent = (xgr[0, 0], xgr[-1, -1], ygr[0, 0], ygr[-1, -1])
iters = [4, 63, 120]
titles = ['before collision', 'during collision', 'after collision']
first = True
for it in iters:
    ue, ve, emag, ub, vb, bmag, col, subset, zdiff, zrange = ues[it], ves[it], emags[it], ubs[it], vbs[it], bmags[it], cols[it], subsets[it], zdiffs[it], zranges[it]
    f = np.sqrt(np.log(1/k)/zrange) * 6
    decayval = np.exp(-np.square(zdiff * f))
    color = np.append(col, decayval.reshape((col.shape[0], 1)), axis=1)
    fact = bmag**0.5 / bmag
    ub *= fact
    vb *= fact
    if first:
        first = False
        quiverobj = axs[i].quiver(xgr, ygr, ub, vb, bmag, cmap='plasma', scale=50)
        qbar = fig.colorbar(quiverobj, ax=axs, location='bottom', label='Magnetic Field Strength (V/m)', aspect=50, fraction=0.05)
    else:
        newq = copyq(quiverobj, axs[i])
        newq.set_UVC(ub, vb, bmag)
    scatterobj = axs[i].scatter(subset[:, 0], subset[:, 1], c=color, s=decayval*5)
    axs[i].set_title(titles[i])
    i += 1

plt.show()
