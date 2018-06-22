#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse.linalg import eigsh

def PCA(data, numPC = 80):
    """
    Return the top principle components of some data. Input (1) the data as a 2D NumPy array, where the observations are along the rows and the data elements of each observation are along the columns, and (2) the number of principle components (numPC) to keep.
    """
    # Number of observations
    M = data.shape[0]
    
    # Mean (not using np.mean for jit reasons)
    mean = data.sum(axis = 0)/M
    data = data - mean
    
    # Covariance (we don't remove the M scaling factor here to try to avoid floating point errors that could make C unsymmetric)
    C = np.dot(data.T, data)
    
    # Compute the top 'numPC' eigenvectors & eigenvalues of the covariance matrix. This uses the scipy.sparse.linalg version of eigh, which happens to be much faster for some reason than the nonsparse version for this case where k << N. Since we didn't remove the M scaling factor in C, the eigenvalues here are scaled by M.
    eigVal, eigVec = eigsh(C, k = numPC, which = 'LM')

    return eigVal[::-1]/M, eigVec[:, ::-1], mean

def rotMat2angle(R):
    """
    Conversion between 3x3 rotation matrix and Euler angles psi, theta, and phi in radians (rotations about the x, y, and z axes, respectively). If the input is 3x3, then the output will return a size-3 array containing psi, theta, and phi. If the input is a size-3 array, then the output will return the 3x3 rotation matrix.
    """
    if R.shape == (3, 3):
        if abs(R[2, 0]) != 1:
            theta = -np.arcsin(R[2, 0])
            psi = np.arctan2(R[2, 1]/np.cos(theta), R[2, 2]/np.cos(theta))
            phi = np.arctan2(R[1, 0]/np.cos(theta), R[0, 0]/np.cos(theta))
        else:
            phi = 0
            if R[2, 0] == -1:
                theta = np.pi/2
                psi = np.arctan2(R[0, 1], R[0, 2])
            else:
                theta = -np.pi/2
                psi = np.arctan2(-R[0, 1], -R[0, 2])
        
        return np.array([psi, theta, phi])
    
    elif R.shape == (3,):
        psi, theta, phi = R
        Rx = np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)], [0, np.sin(psi), np.cos(psi)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        
        return np.dot(Rz, np.dot(Ry, Rx))

def perspectiveTransformKinect(d, inverse = False):
    """
    Transformation between pixel indices (u, v) of depth map to real-world coordinates in mm (x, y) for Kinect v1 depth camera (640x480 resolution). Depth values z are in mm. In the forward direction, go from (u, v, z) to (x, y, z). In the inverse direction, go from (x, y, z) to (u, v, z).
    """
    # Mapping from (x, y, z) to (uz, vz, z)
    real2pixel = np.array([[580.606, 0, 314.758], [0, 580.885, 252.187], [0, 0, 1]])
    
    # Mapping from (uz, vz, z) to (x, y, z)
    pixel2real = np.linalg.inv(real2pixel)
    
    # Mark depth values that are non-zero
    nonZeroZ = d[:, 2] != 0
    
    if not inverse:
        uvz = d[nonZeroZ, :]
        uzvzz = np.c_[np.prod(uvz[:, ::2], axis = 1), np.prod(uvz[:, 1:], axis = 1), uvz[:, 2]]
        xyz = np.dot(pixel2real, uzvzz.T).T
        
        return xyz, nonZeroZ
    
    else:
        xyz = d[nonZeroZ, :]
        uzvzz = np.dot(real2pixel, xyz.T).T
        uvz = np.c_[uzvzz[:, 0] / xyz[:, 2], uzvzz[:, 1] / xyz[:, 2], xyz[:, 2]]
        
        return uvz, nonZeroZ

def sph2cart(el, az):
    """
    Unit sphere elevation and azumuth angles to Cartesian coordinates
    """
    return np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)

def sh9(x, y, z):
    """
    First nine spherical harmonics as functions of Cartesian coordinates
    """
    # h = np.empty((9, x.size))
    # h[0, :] = 1/np.sqrt(4*np.pi) * np.ones(x.size)
    # h[1, :] = np.sqrt(3/(4*np.pi)) * z
    # h[2, :] = np.sqrt(3/(4*np.pi)) * x
    # h[3, :] = np.sqrt(3/(4*np.pi)) * y
    # h[4, :] = 1/2*np.sqrt(5/(4*np.pi)) * (3*np.square(z) - 1)
    # h[5, :] = 3*np.sqrt(5/(12*np.pi)) * x * z
    # h[6 ,:] = 3*np.sqrt(5/(12*np.pi)) * y * z
    # h[7, :] = 3/2*np.sqrt(5/(12*np.pi)) * (np.square(x) - np.square(y))
    # h[8, :] = 3*np.sqrt(5/(12*np.pi)) * x * y
    # return h * np.r_[np.pi, np.repeat(2 * np.pi/ 3, 3), np.repeat(np.pi/ 4, 5)][:, np.newaxis]
    
    # h = np.empty((9, x.size))
    # h[0, :] = np.ones(x.size)
    # h[1, :] = z
    # h[2, :] = x
    # h[3, :] = y
    # h[4, :] = (3 * np.square(z) - 1)
    # h[5, :] = x * z
    # h[6 ,:] = y * z
    # h[7, :] = (np.square(x) - np.square(y))
    # h[8, :] = x * y

    h = np.empty((9, x.size))
    h[0, :] = np.ones(x.size)
    h[1, :] = y
    h[2, :] = z
    h[3, :] = x
    h[4, :] = x * y
    h[5 ,:] = y * z
    h[6, :] = (3 * np.square(z) - 1)
    h[7, :] = x * z
    h[8, :] = (np.square(x) - np.square(y))

    return h