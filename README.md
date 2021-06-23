# Simple-GAN-Experiments-PyTorch

f(x) = x<sup>3</sup> GAN embedding visualization:

![x3 Embedding](/results/GAN_x3_Embedding_v2.JPG)


## Overview

This project is a simple exploration of Generative Adversarial Networks (GANs) embedding some given function in 2D space for ease of visualization and fast model iteration. The goal, as with most GANs, is to train a generator to produce new data points that look like they came from a training dataset. 

Designing the simplest generator and discriminator network possible for the task is extremely helpful in understanding neural networks architectures more deeply and how various hyperparameters affect the adversarial learning process.

## Getting Started

Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```

Now you can change the network architectures, change the function to embedd, hyperparameters, really anything that you want! Running the actual experiment can be done in [Simple_GAN_Experiments.ipynb](https://github.com/shankal17/GAN-Experiments-PyTorch/blob/main/Simple_GAN_Experiments.ipynb).
