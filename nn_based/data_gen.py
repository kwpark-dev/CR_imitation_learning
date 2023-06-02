# import torch 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import multivariate_normal as mv

from utils.kinova_arm import traj_loader, rbf_kernel_1d

path = 'data/kinova_link/artificial_data/'
ground_truth = []

for file in glob('data/kinova_link/raw_data/*.csv')[1:2]:
    name_flag = file.split('/')
    flag = name_flag[-1].split('.')[0] 
    n_stage = 10
    traj, _ = traj_loader(file, header=1, n_cycle=1, mode=False)
    traj = traj.numpy()

    ground_truth.append(traj)
    
    a, b = traj.shape
    sigma = 0.2
    length_scale = 7.4
    mu = np.zeros(b)

    param_t = np.outer(np.ones(n_stage), traj[0]).flatten()
    artificial_data = [param_t]
    for elem in traj[1:]:
        # employ GP prior to generate 'smooth error'
        cov = rbf_kernel_1d(elem, elem, sigma, length_scale)
        stb = 1e-4
        error = mv(mu, cov+stb*np.eye(b)).rvs(n_stage).flatten()
        all_data = np.outer(np.ones(n_stage), elem).flatten()
        artificial_data.append(all_data+0.01*error)

    artificial_data = np.vstack(artificial_data)

    # np.save(path+flag, artificial_data)
    print(flag)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(traj[1], traj[2], traj[3], color='blue', label='ground_truth')
    ax.plot3D(artificial_data[1][:200], artificial_data[2][:200], artificial_data[3][:200], ls='--', color='red', label='artificial traj')
    for i in range(1, n_stage):
        ax.plot3D(artificial_data[1][200*i:200*(i+1)], artificial_data[2][200*i:200*(i+1)], artificial_data[3][200*i:200*(i+1)], ls='--', color='red')
    ax.legend()
    plt.show()




# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(traj[1], traj[2], traj[3])
# ax.plot3D(artificial_data[1], artificial_data[2], artificial_data[3], ls='--', color='red', label='trained traj')
# ax.legend()
# plt.show()