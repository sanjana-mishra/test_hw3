import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
import torch


def plot_points(path):
    ax = plt.figure().add_subplot(projection="3d")
    obj = trimesh.load(path)
    x, y, z = obj.vertices[:, 0], obj.vertices[:, 1], obj.vertices[:, 2]
    mask = obj.colors[:, 1] == 255
    ax.scatter(
        x[mask], y[mask], zs=z[mask], zdir="y", alpha=1, c=obj.colors[mask] / 255
    )
    ax.scatter(
        x[~mask], y[~mask], zs=z[~mask], zdir="y", alpha=0.01, c=obj.colors[~mask] / 255
    )
    plt.show()


def download_data():
    import gdown

    if not os.path.exists("./data"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1EKWU_daQL3pxFkjFUomGs25_qekyfeAd",
            quiet=False,
        )

    if not os.path.exists("./processed"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/175_LtuWh1LknbbMjUumPjGzeSzgQ4ett",
            quiet=False,
        )

def bilinear_interpolation(res, grid, points, grid_type):
    """
    Performs bilinear interpolation of points with respect to a grid.

    Parameters:
        grid (numpy.ndarray): A 2D numpy array representing the grid.
        points (numpy.ndarray): A 2D numpy array of shape (n, 2) representing
            the points to interpolate.

    Returns:
        numpy.ndarray: A 1D numpy array of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()

    # Compute the weights for each of the four points
    w1 = (x2 - x) * (y2 - y)
    w2 = (x - x1) * (y2 - y)
    w3 = (x2 - x) * (y - y1)
    w4 = (x - x1) * (y - y1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res).long()
        id2 = (y1 * res + x2).long()
        id3 = (y2 * res + x1).long()
        id4 = (y2 * res + x2).long()

    elif grid_type == "HASH":
        npts = res**2
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
        else:
            id1 = (x1 + y1 * res).long()
            id2 = (y1 * res + x2).long()
            id3 = (y2 * res + x1).long()
            id4 = (y2 * res + x2).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
    )
    return values[0]


# def trilinear_interpolation(res, grid, points):
#     """
#     Performs trilinear interpolation of points with respect to a 3D grid.

#     Parameters:
#         res (int): Resolution of the grid (assuming a cubic grid).
#         grid (torch.Tensor): A 3D tensor representing the grid.
#         points (torch.Tensor): A 2D tensor of shape (n, 3) representing
#             the points to interpolate.

#     Returns:
#         torch.Tensor: A 1D tensor of shape (n,) representing the interpolated
#             values at the given points.
#     """
#     # Get the dimensions of the grid
#     grid_size, feat_size = grid.shape

#     # Get the x, y, z coordinates of the eight nearest points for each input point
#     x = points[:, 0] * (res - 1)
#     y = points[:, 1] * (res - 1)
#     z = points[:, 2] * (res - 1)

#     x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
#     y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
#     z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

#     x2 = torch.clip(x1 + 1, 0, res - 1).int()
#     y2 = torch.clip(y1 + 1, 0, res - 1).int()
#     z2 = torch.clip(z1 + 1, 0, res - 1).int()

#     # Compute the weights for each of the eight points
#     w1 = (x2 - x) * (y2 - y) * (z2 - z)
#     w2 = (x - x1) * (y2 - y) * (z2 - z)
#     w3 = (x2 - x) * (y - y1) * (z2 - z)
#     w4 = (x - x1) * (y - y1) * (z2 - z)
#     w5 = (x2 - x) * (y2 - y) * (z - z1)
#     w6 = (x - x1) * (y2 - y) * (z - z1)
#     w7 = (x2 - x) * (y - y1) * (z - z1)
#     w8 = (x - x1) * (y - y1) * (z - z1)

#     # Interpolate the values for each point
#     values = (
#         w1 * grid[x1, y1, z1]
#         + w2 * grid[x2, y1, z1]
#         + w3 * grid[x1, y2, z1]
#         + w4 * grid[x2, y2, z1]
#         + w5 * grid[x1, y1, z2]
#         + w6 * grid[x2, y1, z2]
#         + w7 * grid[x1, y2, z2]
#         + w8 * grid[x2, y2, z2]
#     )

#     return values

def trilinear_interpolation(res, grid, points, grid_type):
    """
    Performs trilinear interpolation of points with respect to a grid.

    Parameters:
        grid (numpy.ndarray): A 3D numpy array representing the grid.
        points (numpy.ndarray): A 2D numpy array of shape (n, 3) representing
            the points to interpolate.

    Returns:
        numpy.ndarray: A 1D numpy array of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861, 1597334677]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x, y, and z coordinates of the eight nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)
    z = points[:, :, 2] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
    z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()
    z2 = torch.clip(z1 + 1, 0, res - 1).int()

    # Compute the weights for each of the eight points
    w1 = (x2 - x) * (y2 - y) * (z2 - z)
    w2 = (x - x1) * (y2 - y) * (z2 - z)
    w3 = (x2 - x) * (y - y1) * (z2 - z)
    w4 = (x - x1) * (y - y1) * (z2 - z)
    w5 = (x2 - x) * (y2 - y) * (z - z1)
    w6 = (x - x1) * (y2 - y) * (z - z1)
    w7 = (x2 - x) * (y - y1) * (z - z1)
    w8 = (x - x1) * (y - y1) * (z - z1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res + z1 * res * res).long()
        id2 = (x2 + y1 * res + z1 * res * res).long()
        id3 = (x1 + y2 * res + z1 * res * res).long()
        id4 = (x2 + y2 * res + z1 * res * res).long()
        id5 = (x1 + y1 * res + z2 * res * res).long()
        id6 = (x2 + y1 * res + z2 * res * res).long()
        id7 = (x1 + y2 * res + z2 * res * res).long()
        id8 = (x2 + y2 * res + z2 * res * res).long()

    elif grid_type == "HASH":
        npts = res**3
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id5 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id6 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id7 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id8 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
        else:
            id1 = (x1 + y1 * res + z1 * res * res).long()
            id2 = (x2 + y1 * res + z1 * res * res).long()
            id3 = (x1 + y2 * res + z1 * res * res).long()
            id4 = (x2 + y2 * res + z1 * res * res).long()
            id5 = (x1 + y1 * res + z2 * res * res).long()
            id6 = (x2 + y1 * res + z2 * res * res).long()
            id7 = (x1 + y2 * res + z2 * res * res).long()
            id8 = (x2 + y2 * res + z2 * res * res).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
        + torch.einsum("ab,abc->abc", w5, grid[(id5).long()])
        + torch.einsum("ab,abc->abc", w6, grid[(id6).long()])
        + torch.einsum("ab,abc->abc", w7, grid[(id7).long()])
        + torch.einsum("ab,abc->abc", w8, grid[(id8).long()])
    )
    return values[0]