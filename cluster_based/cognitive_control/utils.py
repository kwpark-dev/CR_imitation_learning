import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot3d(x, y, z, title):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid()
    ax.set_title(title)

    plt.show()


def make_ellipses(means, covs, ax):
    '''
    Function to compute and plot the ellipses from each component's covariance matrix.
    Modification of code written by Harald in BIGP.
    '''
    # colors = ['red', 'green', 'blue']

    for n, color in enumerate(means):
        covariances = covs[n]#[:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)

        if n == 0:
            ell = mpl.patches.Ellipse(means[n, :2], 2*v[0], 2*v[1],
                                      180 + angle, color='cyan', label='2$\sigma$ clusters')
        else:
            ell = mpl.patches.Ellipse(means[n, :2], 2*v[0], 2*v[1],
                                    180 + angle, color='cyan')

        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def data_generator(n):
    t = np.linspace(0, 10, n)
    
    x = np.sin(t)**3
    y = 0.01 * t**2
    pose = np.arctan2(y, x)
    error = np.random.random((3, n)) * 0.1

    data = np.array([x, y, pose]) + error

    return data


def gen_trajectory1d(n):
    t = np.linspace(0, 10, n)
    
    x = np.sin(t)**3
    error = np.random.random(n) * 0.1

    x += error
    
    data = np.array([t, x])

    return data


def gen_trajectory2d(n):
    t = np.linspace(0, 10, n)
    
    x = np.sin(t)**3
    y = 0.01 * t**2
    errorx = np.random.random(n) * 0.1
    errory = np.random.random(n) * 0.1

    x += errorx
    y += errory
    
    data = np.array([t, x, y])

    return data
