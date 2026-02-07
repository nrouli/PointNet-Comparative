import torch
import torch.nn as nn
import scipy.linalg
import numpy as np

import random
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from models import PointNetPP, CGAPointNetPP, PointCMLP

EPSILON = 1e-8


def identity(x):
    # needed in this format to save the model properly
    return x


def build_mlgp(input_shape=(4, 3), output_dim=10, hidden_layer_sizes=[4], bias=False, activation=identity):
    # Multilayer Geometric Perceptron
    print('\nmodel: MLGP')
    model = PointCMLP(input_shape, output_dim, hidden_layer_sizes, activation, bias, version=1)
    return model


def build_point_net_pp(output_dim=10, activation=nn.functional.relu, dropout=0.2, version=0):
    print('\nmodel: PointNet++')
    model = PointNetPP(output_dim, activation, dropout, version=version)
    return model



def build_cgapoint_net_pp(output_dim=10, activation=identity):
    print('\nmodel: PointNet++')
    model = CGAPointNetPP(output_dim, activation)
    return model



def score(y, t):
    return torch.mean((torch.argmax(y, axis=1) == t).double()).item()



def save_checkpoint(state, save_dir='pretrained_models'):
    torch.save(state, save_dir+'/'+state['name']+'.tar')



def random_rotation_matrix(low=[0.0], high=[1.0]):
    """
    Inspired by
    https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/tensorfieldnetworks/utils.py

    Generates a random 3D rotation matrix.

    Args:
        low, high:  intergers, or floats, or tuples/lists;
                    the lower and upper bounds of the random rotation angle,
                    specified as fractions of 2*pi;
                    in case of tuples/lists, the intervals are formed
                    by taking the bounds from low and high pair-wise, e.g., 
                    low=[0.0, 1/4], high=[1/8, 1.0] corresponds to 
                    [0, 2*pi/8) U [2*pi/4, 2*pi) = [0, pi/4) U [pi/2, 2*pi).
                    The angle is drawn from the distribution over the joint interval.
    Returns:
        Random rotation matrix.
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON

    theta = 2 * np.pi * np.random.uniform(low, high)
    theta = np.random.choice(np.atleast_1d(theta))

    return rotation_matrix(axis, theta)



def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))



def get_model_net_data(train_size=3991, test_size=908,
                       distortion=None, n_points=1024, force_reload=False, class_size=10, root='data/'):
    
    pre_transform = []
    transform=[]
    
    pre_transform.append(T.SamplePoints(n_points))
    pre_transform.extend([T.Center(), T.NormalizeScale()])
    
    
    if distortion:
        transform.append(T.RandomJitter(distortion))
    
    pre_transform = T.Compose(pre_transform)
    transform = T.Compose(transform) if transform else None

    train_dataset = ModelNet(root=root, name='10', train=True, pre_transform=pre_transform, transform=transform, force_reload=force_reload)
    test_dataset = ModelNet(root=root, name='10', train=False, pre_transform=pre_transform, transform=None, force_reload=force_reload)

    train_dataset = [data for data in train_dataset if data.y.item() in range(class_size)]
    test_dataset = [data for data in test_dataset if data.y.item() in range(class_size)]
    
    print(f"Loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    #shuffle the entire dataset:
    random.shuffle(train_dataset), random.shuffle(test_dataset)

    Xtrain = torch.stack([data.pos for data in train_dataset[:train_size]])
    Ytrain = torch.tensor([data.y.item() for data in train_dataset[:train_size]])

    Xtest = torch.stack([data.pos for data in test_dataset[:test_size]])
    Ytest = torch.tensor([data.y.item() for data in test_dataset[:test_size]])


    return (Xtrain.float(), Ytrain.long()), (Xtest.float(), Ytest.long())