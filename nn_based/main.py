import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from utils.kinova_arm import traj_loader
from architecture.controller_1d import PhysicsNet

if "__main__" == __name__:
    
    path = 'data/kinova_link/artificial_data/*'
    n_stages = 10
    # x_traj, mean_traj = traj_loader(path, mode=False)
    for file in glob(path):
        file_name = file.split('/')
        link = file_name[-1].split('.')[0]
        x_traj = np.load(file)
        x_traj = torch.from_numpy(x_traj).float()

        u_coords = x_traj[1:].T
        boundary = np.array([0, -1])
        x_lb, x_ub = x_traj[0][boundary]
        
        input_node_phy = torch.tensor([1])
        hidden_node_phy = torch.tensor([6, 12, 24, 12])
        output_node_phy = torch.tensor([6])
        
        physics_analyzer = PhysicsNet(input_node_phy, hidden_node_phy, output_node_phy)
        physics_finder = optim.Adam(physics_analyzer.parameters())
        # solution_loss_func = nn.MSELoss()
        solution_loss_func = nn.HuberLoss()
        # solution_loss_func2 = nn.MSELoss()
        
        
        solution_loss_train = []
        # learning_physics_train = []

        physics_epochs = 30000
        for i in tqdm(range(physics_epochs)):
            physics_finder.zero_grad()
                
            u_est = physics_analyzer(x_traj[0][:, None])
            loss = solution_loss_func(u_est, u_coords)
            solution_loss_train.append(loss.item())

            loss.backward()
            physics_finder.step()


        plt.plot(solution_loss_train)
        step = (x_ub.item()-x_lb.item())/300
        time_domain = torch.from_numpy(np.arange(x_lb.item(), x_ub.item()+step, step)).float()
        trained_traj = physics_analyzer(time_domain[:, None]).detach()
        data = np.hstack((time_domain[:, None], trained_traj))

        # np.save('data/kinova_link/prediction/'+link, data)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x_traj[1][:200], x_traj[2][:200], x_traj[3][:200], ls='--', color='red', label='artificial traj')
        for i in range(1, n_stages):
            ax.plot3D(x_traj[1][200*i:200*(i+1)], x_traj[2][200*i:200*(i+1)], x_traj[3][200*i:200*(i+1)], ls='--', color='red')
        # ax.plot3D(mean_traj.T[1], mean_traj.T[2], mean_traj.T[3], color='black', label='artificial traj')
        ax.plot3D(trained_traj.T[0], trained_traj.T[1], trained_traj.T[2], color='black', label='trained traj')
        ax.legend()
        ax.set_title(link)
        plt.show()

        # print(data)
        # np.save('test', data)
    # plt.plot(time_domain, trained_traj, 'r', label='trained_traj')
    # plt.plot(mean_traj[0], mean_traj[1], ls='--', color='orange', label='mean_traj')
    # plt.scatter(x_traj[0], x_traj[1], s=4, label='trajectories')
    # plt.legend()
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(mean_traj.T[1], mean_traj.T[2], mean_traj.T[3], color='black', label='mean traj')
    # ax.plot3D(trained_traj.T[0], trained_traj.T[1], trained_traj.T[2], ls='--', color='red', label='trained traj')
    # ax.legend()
    # plt.show()
