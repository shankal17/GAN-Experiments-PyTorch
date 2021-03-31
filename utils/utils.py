import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_ground_truth(func, num_points):
    X1 = np.random.rand(num_points) - 0.5
    X2 = func(X1)
    X1 = X1.reshape(num_points, 1)
    X2 = X2.reshape(num_points, 1)
    X = np.hstack((X1, X2))
    X = torch.from_numpy(X).float()
    y = np.ones((num_points, 1))
    y = torch.from_numpy(y).float()
    return X, y

def run_generator(generator, num_points):
    latent_input = np.random.randn(num_points, generator.latent_space_dim)
    latent_input = torch.from_numpy(latent_input).float()
    X_output = generator(latent_input)
    y = np.zeros((num_points, 1))
    y = torch.from_numpy(y).float()
    return X_output, y
