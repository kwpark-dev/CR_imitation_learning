import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from cognitive_control.utils import make_ellipses, gen_trajectory1d, gen_trajectory2d
from cognitive_control.clustering import kmeans, MixtureGaussian


if "__main__" == __name__:

    # collect 10000 sequential data which has 51 sequences
    # has 2 fixed points, start & end (defined variables below)
    stages = 30
    N = 51
    
    t = np.linspace(0, 10, N)
    x = np.sin(t)**3
    y = 0.01 * t**2
    
    data = np.concatenate([gen_trajectory1d(N) for _ in range(stages)], axis=1)
    
    n_cluster = 10
    data_dim = 2 # time + measurements
    init_centers, init_covs, init_weights = kmeans(data.T, n_cluster, data_dim)
    gmm = MixtureGaussian(data.T, init_centers, init_covs, init_weights)
    centers, covs, weights = gmm.fit()
    sample = gmm.sample(N)
    filtered = savgol_filter(sample.T, 5, 3)
    
    fig, axes = plt.subplots(2,2, figsize=(12, 12))
    make_ellipses(init_centers, init_covs, axes[0][0])
    axes[0][0].plot(t, x, 'black', label='averaged_trajectory')
    axes[0][0].scatter(data[0], data[1], s=2)
    axes[0][0].grid()
    axes[0][0].set_xlabel('param t')
    axes[0][0].set_ylabel('x(t)')
    axes[0][0].legend()
    axes[0][0].set_title('K-means initialization')

    make_ellipses(centers, covs, axes[0][1])
    axes[0][1].plot(t, x, 'black', label='averaged_trajectory')
    axes[0][1].grid()
    axes[0][1].set_xlabel('param t')
    axes[0][1].set_ylabel('x(t)')
    axes[0][1].legend()
    axes[0][1].set_title('EM clustering')

    make_ellipses(centers, covs, axes[1][0])
    axes[1][0].plot(t, x, 'black', label='averaged_trajectory')
    axes[1][0].plot(t, sample.T[1], 'red', label='sampled_trajectory')
    axes[1][0].grid()
    axes[1][0].set_xlabel('param t')
    axes[1][0].set_ylabel('x(t)')
    axes[1][0].legend()
    axes[1][0].set_title('Sampled sequence')

    make_ellipses(centers, covs, axes[1][1])
    axes[1][1].plot(t, x, 'black', label='averaged_trajectory')
    axes[1][1].plot(t, filtered[1], 'blue', label='filtered_trajectory')
    axes[1][1].grid()
    axes[1][1].set_xlabel('param t')
    axes[1][1].set_ylabel('x(t)')
    axes[1][1].legend()
    axes[1][1].set_title('Sample sesquence after filtering')
    
    plt.show()