import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_3d_grid(shape):
    """
    Create a 3D grid of points.
    """
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    z = np.linspace(0, shape[2] - 1, shape[2])
    grid = np.meshgrid(x, y, z, indexing='ij')
    return grid

def plot_deformation_field(deformation_field, step=5):
    """
    Plot the 3D deformation field.
    """
    shape = deformation_field.shape[:-1]
    grid = create_3d_grid(shape)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract grid points and deformation vectors
    x, y, z = grid
    u, v, w = deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2]
    
    # Plot the deformation field
    ax.quiver(x[::step, ::step, ::step], y[::step, ::step, ::step], z[::step, ::step, ::step],
              u[::step, ::step, ::step], v[::step, ::step, ::step], w[::step, ::step, ::step],
              length=1, normalize=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Deformation Field')
    plt.show()

# Example usage
# Assuming deformation_field is a numpy array of shape (D, H, W, 3)
# where D, H, W are the dimensions of the volume and 3 represents the deformation vectors
deformation_field = np.random.rand(32, 32, 32, 3)  # Replace with your actual deformation field
plot_deformation_field(deformation_field)