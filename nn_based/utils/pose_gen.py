import numpy as np
import torch


def arm_1d_bc_data(n_sample):
    t = np.random.uniform(0, 4, (n_sample, 1))
    x = np.zeros_like(t)
    u = np.zeros_like(t)

    return torch.from_numpy(t).float(), torch.from_numpy(x).float(), torch.from_numpy(u).float()


def collocation_data(n):
    # 0 <= t <= 4
    # -1 <= x <= 3
    x = np.linspace(-1, 3, n)
    t = np.linspace(0, 4, n)

    X, T = np.meshgrid(x, t)
    X = X.flatten()
    T = T.flatten()

    return torch.from_numpy(T[:, None]).float(), torch.from_numpy(X[:, None]).float()


def arm_1d_ic_cnt_data(var, n_sample):
    # particular time: 0, 2, 4
    r = 1
    error = np.random.normal(0, var, (n_sample, 3))
    angle = np.ones_like(error)*np.array([np.pi/3, np.pi/4, -np.pi/5]) + error

    node01 = np.zeros((n_sample, 2))
    node02 = node01 + np.array([r*np.cos(angle[:,0]), r*np.sin(angle[:,0])]).T
    node03 = node02 + np.array([r*np.cos(angle[:,1]), r*np.sin(angle[:,1])]).T
    node04 = node03 + np.array([r*np.cos(angle[:,2]), r*np.sin(angle[:,2])]).T
    
    pose_01 = np.concatenate([node01, node02, node03, node04])
    t0 = np.zeros_like(pose_01[:, 0][:,None])
    t1 = np.ones_like(pose_01[:, 0][:,None])*2
    t2 = np.ones_like(pose_01[:, 0][:,None])*4

    T = np.concatenate([t1,t2])

    error2 = np.random.normal(0, var, (n_sample, 3))
    angle2 = np.ones_like(error)*np.array([np.pi/5, -np.pi/7, -np.pi/3]) + error2

    node02 = node01 + np.array([r*np.cos(angle2[:,0]), r*np.sin(angle2[:,0])]).T
    node03 = node02 + np.array([r*np.cos(angle2[:,1]), r*np.sin(angle2[:,1])]).T
    node04 = node03 + np.array([r*np.cos(angle2[:,2]), r*np.sin(angle2[:,2])]).T

    pose_02 = np.concatenate([node01, node02, node03, node04])

    error3 = np.random.normal(0, var, (n_sample, 3))
    angle3 = np.ones_like(error)*np.array([np.pi/1.5, np.pi/4, -np.pi/2]) + error3

    node02 = node01 + np.array([r*np.cos(angle3[:,0]), r*np.sin(angle3[:,0])]).T
    node03 = node02 + np.array([r*np.cos(angle3[:,1]), r*np.sin(angle3[:,1])]).T
    node04 = node03 + np.array([r*np.cos(angle3[:,2]), r*np.sin(angle3[:,2])]).T

    pose_03 = np.concatenate([node01, node02, node03, node04])

    pose = np.concatenate([pose_02, pose_03])

    return torch.from_numpy(t0).float(), torch.from_numpy(pose_01[:, 0][:, None]).float(), \
           torch.from_numpy(pose_01[:, 1][:, None]).float(), torch.from_numpy(T).float(), \
           torch.from_numpy(pose[:, 0][:, None]).float(), torch.from_numpy(pose[:, 1][:, None]).float()