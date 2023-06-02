import torch
import numpy as np
from glob import glob


def rbf_kernel_1d(x1, x2, eta, width):
    tmp_sq = np.square(x1)[:, np.newaxis] + np.square(x2)[np.newaxis] - 2 * np.outer(x1, x2)
    cov = eta * np.exp(-0.5 * tmp_sq / width / width)
    
    return cov


def rbf_kernel(x1, x2, sigma, width): 
    n = len(x1)
    m = len(x2)
    var = np.array([])
    
    for i in x1:
        exponent = np.linalg.norm((x2-i)/width, axis=1)**2
        var = np.concatenate((var,sigma*np.exp(-0.5*exponent)))
    
    return var.reshape(n,m)


def traj_loader(path, header=0, n_cycle=30, mode=True): 

    collection = []
    mean_traj = 0
    cut_off_size = 10000
    scale = 0.5
    t = np.arange(1, cut_off_size, 50, dtype=int)
    param_t = np.arange(0, len(t)*scale, scale, dtype=float)

    for sample in glob(path):
        data = np.genfromtxt(sample, delimiter=',' , skip_header=header)
        masked_data = data[t]
        mean_traj += masked_data[:, 1:]/n_cycle
        rearr_data = np.hstack((param_t[np.newaxis].T, masked_data[:, 1:]))
        collection.append(rearr_data)

    collection = np.array(collection)
    cluster = np.concatenate(collection, axis=0).T
    
    mean_traj = np.hstack((param_t[:, np.newaxis], mean_traj))

    if mode:
        return torch.from_numpy(cluster[:2]).float(), torch.from_numpy(mean_traj.T[:2]).float()
        
    else :
        return torch.from_numpy(cluster).float(), torch.from_numpy(mean_traj).float()


def boundary_data(mean_traj, boundary_idx):
    x_lb, x_ub = mean_traj[0][boundary_idx]
    u_lb, u_ub = mean_traj[1][boundary_idx]

    t = np.random.choice(np.arange(0, 1.1, 0.1), (200, 1))
    idx = np.arange(len(t))
    np.random.shuffle(idx)
    
    x = torch.concat((torch.ones((100,1))*x_lb, torch.ones((100,1))*x_ub)).float()
    u = torch.concat((torch.ones((100,1))*u_lb, torch.ones((100,1))*u_ub)).float()

    return torch.from_numpy(t).float(), x[idx], u[idx]


def collocation_data(mean_traj, boundary_idx):
    x_lb, x_ub = mean_traj[0][boundary_idx]
    
    t = np.linspace(0, 1, 20)
    x = np.linspace(x_lb, x_ub, 20)

    T, X = np.meshgrid(t, x)
    T = T.flatten()[:, None]
    X = X.flatten()[:, None]

    return T, X