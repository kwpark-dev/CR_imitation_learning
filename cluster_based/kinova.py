import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from cognitive_control.utils import make_ellipses
from cognitive_control.clustering import kmeans, MixtureGaussian
from scipy.signal import savgol_filter


if "__main__" == __name__:

    collection = []
    mean_traj = 0
    cut_off_size = 9500
    scale = 0.01
    t = np.arange(1, cut_off_size, 50, dtype=int)
    param_t = np.arange(0, len(t)*scale, scale, dtype=float)
    
    for sample in glob('data/*'):
        data = np.genfromtxt(sample, delimiter=',')
        masked_data = data[t]
        
        mean_traj += masked_data[:, 1:4]/30
        rearr_data = np.hstack((param_t[np.newaxis].T, masked_data[:, 1:4]))
        collection.append(rearr_data)

    collection = np.array(collection)
    cluster = np.concatenate(collection, axis=0).T
    x_data = cluster[:2]

    # print(x_data)
    # plt.plot(t, mean_traj.T[0], 'r')
    # plt.scatter(x_data[0], x_data[1])
    # plt.show()
    N = len(t)
    # n_cluster = 7
    set_of_clusters = [3, 4, 5, 6]
    data_dim = 4 # time + measurements
    my_traj = []
    for n_cluster in set_of_clusters:
        init_centers, init_covs, init_weights = kmeans(cluster.T, n_cluster, data_dim)

        print(init_centers, init_covs)
        
        gmm = MixtureGaussian(cluster.T, init_centers, init_covs, init_weights, eps=0.005)
        centers, covs, weights = gmm.fit()
        
        print(centers, covs)
        
        sample = gmm.sample(N)
        filtered = savgol_filter(sample.T, 21, 3, mode='nearest')
        my_traj.append(filtered)
    
    # fig, axes = plt.subplots(2,2, figsize=(12, 12))

    # plt.suptitle('2D Visualization')
    # make_ellipses(init_centers, init_covs, axes[0][0])
    # axes[0][0].plot(param_t, mean_traj.T[0], 'red', label='averaged_trajectory')
    # axes[0][0].scatter(x_data[0], x_data[1], s=2, color='blue', label='collected_trajecotry')
    # axes[0][0].grid()
    # axes[0][0].set_xlabel('param t')
    # axes[0][0].set_ylabel('x(t)')
    # axes[0][0].legend()
    # axes[0][0].set_title('K-means initialization')

    # make_ellipses(centers, covs, axes[0][1])
    # axes[0][1].plot(param_t, mean_traj.T[0], 'red', label='averaged_trajectory')
    # axes[0][1].scatter(x_data[0], x_data[1], s=2, color='blue', label='collected_trajecotry')
    # axes[0][1].grid()
    # axes[0][1].set_xlabel('param t')
    # axes[0][1].set_ylabel('x(t)')
    # axes[0][1].legend()
    # axes[0][1].set_title('EM clustering')

    # make_ellipses(centers, covs, axes[1][0])
    # axes[1][0].plot(param_t, mean_traj.T[0], 'red', label='averaged_trajectory')
    # axes[1][0].plot(param_t, sample.T[1], 'orange', label='sampled_trajectory')
    # axes[1][0].grid()
    # axes[1][0].set_xlabel('param t')
    # axes[1][0].set_ylabel('x(t)')
    # axes[1][0].legend()
    # axes[1][0].set_title('Sampled sequence')

    # make_ellipses(centers, covs, axes[1][1])
    # axes[1][1].plot(param_t, mean_traj.T[0], 'red', label='averaged_trajectory')
    # axes[1][1].plot(param_t, filtered[1], 'blue', label='filtered_trajectory')
    # axes[1][1].grid()
    # axes[1][1].set_xlabel('param t')
    # axes[1][1].set_ylabel('x(t)')
    # axes[1][1].legend()
    # axes[1][1].set_title('Sample sesquence after filtering')
    
    # plt.show()


    colors = ['red', 'cyan', 'orange', 'purple']
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(mean_traj.T[0], mean_traj.T[1], mean_traj.T[2], color='black', label='mean trajectory')
    
    for idx, d in enumerate(my_traj):
        ax.scatter3D(d[1], d[2], d[3], color=colors[idx], label=str(idx+3)+'_clusters trajectory')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid()
    ax.set_title('trajectory of end effector')
    ax.legend()

    plt.show()