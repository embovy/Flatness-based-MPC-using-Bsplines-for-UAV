import numpy as np 
import casadi as cd

# Integration via Euler
def Euler(x0, dx, Ts, N):
    xx = x0
    x_int = np.zeros((12,N))
    for i in range(N):
        x_int[:,i] = xx
        xx = xx + Ts/N*dx[:,i]
    return x_int

# Numerical derivation
def derivative(x, dt, method='central'):
    """
    Compute the first derivate of a function given by a discrete set of data points x, equidistant in time

    Args:
        x (array_like): array with data points [x0, x1, ..., xn]
        dt (float): constant time step
        method (string):    central == central difference 
                            forward == forward Euler
                            backward == backward Euler

    Returns:
        dx (array_like): array with derivatives [dx0, dx1, ..., dxn]
    """

    dx = np.zeros((x.shape[0], x.shape[1]))
    dx[:,0] = (x[:,1] - x[:,0])/dt
    dx[:,-1] = (x[:,-1] - x[:,-2])/dt

    if method == 'central':
        for i in range(1, x.shape[1] - 1):
            dx[:,i] = (x[:,i+1] - x[:,i-1])/(2*dt)
        return dx

    if method == 'forward':
        for i in range(1, x.shape[1] - 1):
            dx[:,i] = (x[:,i+1] - x[:,i])/dt
        return dx

    if method == 'backward': 
        for i in range(1, x.shape[1] - 1):
            dx[:,i] = (x[:,i] - x[:,i-1])/dt
        return dx

# 3D Rotation matrices
def Rotx(phi, package='numpy'):
    """
    Args:
        phi (float): rotation angle

    Return:
        Rotx (array_like): rotation matrix around the x-axis
    """
    if package == 'casadi':
        return cd.MX([[1, 0, 0],
                    [0, cd.cos(phi), -cd.sin(phi)],
                    [0, cd.sin(phi), cd.cos(phi)]])
    
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Roty(theta, package='numpy'):
    """
    Args:
        theta (float): rotation angle

    Return:
        Roty (array_like): rotation matrix around the y-axis
    """
    if package == 'casadi':
        return cd.MX([[cd.cos(theta), 0, cd.sin(theta)],
                    [0, 1, 0],
                    [-cd.sin(theta), 0, cd.cos(theta)]])

    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rotz(psi, package='numpy'):
    """
    Args:
        psi (float): rotation angle

    Return:
        Rotz (array_like): rotation matrix around the z-axis
    """
    if package == 'casadi':
        return cd.MX([[cd.cos(psi), -cd.sin(psi), 0],
                    [cd.sin(psi), cd.cos(psi), 0],
                    [0, 0, 1]])

    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])