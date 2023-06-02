import torch
import torch.nn as nn
from torch.autograd import grad



def physics_informed_network(t, x, model, params):
    
    params = params.detach()
    t.requires_grad = True
    x.requires_grad = True

    u = model(t, x)

    ut = grad(outputs=u.sum(), inputs=t, create_graph=True)[0]
    utt = grad(outputs=ut.sum(), inputs=t, create_graph=True)[0]
    ux = grad(outputs=u.sum(), inputs=x, create_graph=True)[0]
    uxx = grad(outputs=ux.sum(), inputs=x, create_graph=True)[0]
    utx = grad(outputs=ut.sum(), inputs=x, create_graph=True)[0]

    f = params[:,0]*utt + params[:,1]*ut + params[:,2]*ux + params[:,3]*uxx + params[:,4]*utx + params[:,5]

    return f


def data_driven_physics(t, params, model):
    
    t.requires_grad = True

    u = model(t)
    ut = grad(outputs=u.sum(), inputs=t, create_graph=True)[0]
    utt = grad(outputs=ut.sum(), inputs=t, create_graph=True)[0]
    
    f = params[:,0]*utt + params[:,1]*ut + params[:,2]*u + params[:,3]

    return f


class PhysicsNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(PhysicsNet, self).__init__()

        layer_info = torch.cat((input, hidden, output))
        self.model = self.__gen_model(layer_info)
        

    def __gen_model(self, layer_info):
        layers = []

        for i in range(len(layer_info)-1):

            if i == len(layer_info)-2:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                
            else:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                # layers.append(nn.BatchNorm1d(layer_info[i+1]))

        return nn.Sequential(*layers)


    def forward(self, t):
        u = self.model(t)
        
        return u


class SkyNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(SkyNet, self).__init__()

        layer_info = torch.cat((input, hidden, output))
        self.model = self.__gen_model(layer_info)
        

    def __gen_model(self, layer_info):
        layers = []

        for i in range(len(layer_info)-1):

            if i == len(layer_info)-2:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                
            else:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                # layers.append(nn.BatchNorm1d(layer_info[i+1]))

        return nn.Sequential(*layers)


    def forward(self, t, x):
        grid = torch.concat([t,x], axis=1)
        u = self.model(grid)
        
        return u