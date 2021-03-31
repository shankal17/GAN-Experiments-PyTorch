import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.linear_leakyrelu_stack = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.LeakyReLU(0.1),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear_leakyrelu_stack(x)
        return out

class Generator(nn.Module):
    def __init__(self, latent_space_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.output_dim = output_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(latent_space_dim, 15),
            nn.LeakyReLU(0.3),
            nn.Linear(15, output_dim))

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
