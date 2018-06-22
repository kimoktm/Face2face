#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains helper functions that take the derivative of certain functions in the rendering pipeline.
"""

import numpy as np
from sklearn.preprocessing import normalize

def dR_dpsi(angles):
    """Returns the derivative of the rotation matrix with respect to the x-axis rotation angle.
    
    Args:
        angles (ndarray (3,)): Euler angles
    
    Returns:
        ndarray, (3, 3): derivative of rotation matrix with respect to psi
    """
    psi, theta, phi = angles
    return np.array([[0, np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi), np.cos(psi)*np.sin(phi) - np.sin(psi)*np.sin(theta)*np.cos(phi)], [0, -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.cos(phi) - np.sin(psi)*np.sin(theta)*np.sin(phi)], [0, np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(theta)]])

def dR_dtheta(angles):
    """Returns the derivative of the rotation matrix with respect to the y-axis rotation angle.
    
    Args:
        angles (ndarray (3,)): Euler angles
    
    Returns:
        ndarray, (3, 3): derivative of rotation matrix with respect to theta
    """
    psi, theta, phi = angles
    return np.array([[-np.sin(theta)*np.cos(phi), np.sin(psi)*np.cos(theta)*np.cos(phi), np.cos(psi)*np.cos(theta)*np.cos(phi)], [-np.sin(theta)*np.sin(phi), np.sin(psi)*np.cos(theta)*np.sin(phi), np.cos(psi)*np.cos(theta)*np.sin(phi)], [-np.cos(theta), -np.sin(psi)*np.sin(theta), -np.cos(psi)*np.sin(theta)]])

def dR_dphi(angles):
    """Returns the derivative of the rotation matrix with respect to the z-axis rotation angle.
    
    Args:
        angles (ndarray (3,)): Euler angles
    
    Returns:
        ndarray, (3, 3): derivative of rotation matrix with respect to phi
    """
    psi, theta, phi = angles
    return np.array([[-np.cos(theta)*np.sin(phi), -np.cos(psi)*np.cos(phi) - np.sin(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.cos(phi) - np.cos(psi)*np.sin(theta)*np.sin(phi)], [np.cos(theta)*np.cos(phi), -np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi), np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)], [0, 0, 0]])

def dR_sh(x, y, z, dR_x, dR_y, dR_z):
    """Returns the derivative of the spherical harmonics
    
    Args:
        x, y, z (ndarray): normals in X, Y, Z
        dR_x, dR_y, dR_z (ndarray): normals derivatives in X, Y, Z

    Returns:
        ndarray, (9, ndarray): derivative of spherical harmonics
    """

    dR_h = np.empty((9, x.size, dR_x.shape[1]))

    # dR_h[0, :] = np.zeros((x.size, dR_x.shape[1]))
    # dR_h[1, :] = dR_z
    # dR_h[2, :] = dR_x
    # dR_h[3, :] = dR_y
    # dR_h[4, :] = 6 * np.multiply(dR_z, z[:, np.newaxis])
    # dR_h[5, :] = np.multiply(dR_z, x[:, np.newaxis]) + np.multiply(dR_x, z[:, np.newaxis])
    # dR_h[6 ,:] = np.multiply(dR_z, y[:, np.newaxis]) + np.multiply(dR_y, z[:, np.newaxis])
    # dR_h[7, :] = 2 * (np.multiply(dR_x, x[:, np.newaxis]) - np.multiply(dR_y, y[:, np.newaxis]))
    # dR_h[8, :] = np.multiply(dR_y, x[:, np.newaxis]) + np.multiply(dR_x, y[:, np.newaxis])

    dR_h[0, :] = np.zeros((x.size, dR_x.shape[1]))
    dR_h[1, :] = dR_y
    dR_h[2, :] = dR_z
    dR_h[3, :] = dR_x
    dR_h[4, :] = np.multiply(dR_y, x[:, np.newaxis]) + np.multiply(dR_x, y[:, np.newaxis])
    dR_h[5 ,:] = np.multiply(dR_z, y[:, np.newaxis]) + np.multiply(dR_y, z[:, np.newaxis])
    dR_h[6, :] = 6 * np.multiply(dR_z, z[:, np.newaxis])
    dR_h[7, :] = np.multiply(dR_z, x[:, np.newaxis]) + np.multiply(dR_x, z[:, np.newaxis])
    dR_h[8, :] = 2 * (np.multiply(dR_x, x[:, np.newaxis]) - np.multiply(dR_y, y[:, np.newaxis]))

    return dR_h

def dR_normal(vertexCoord, model, dR_vertex):
    """Calculates the per-vertex normal vectors for a model given shape coefficients.
    
    Args:
        vertexCoord (ndarray): Vertex coordinates for the 3DMM, (3, numVertices)
        model (MeshModel): 3DMM MeshModel class object
    
    Returns:
        ndarray: Per-vertex normal vectors
    """

    if len(dR_vertex.shape) is 2:
        dR_vertex = dR_vertex[:, :, np.newaxis]

    a = vertexCoord[:, model.face[:, 0]] - vertexCoord[:, model.face[:, 1]]
    b = vertexCoord[:, model.face[:, 0]] - vertexCoord[:, model.face[:, 2]]

    dR_a = dR_vertex[:, model.face[:, 0]] - dR_vertex[:, model.face[:, 1]]
    dR_b = dR_vertex[:, model.face[:, 0]] - dR_vertex[:, model.face[:, 2]]
    
    dR_faceNorm = np.empty((a.shape[1], a.shape[0], dR_a.shape[2]))
    for v in range(0, dR_a.shape[2]):
        dR_faceNorm[:, :, v] = np.cross(dR_a[:,:,v], b, axisa = 0, axisb = 0) + np.cross(a, dR_b[:,:,v], axisa = 0, axisb = 0)

    dR_vNorm = np.array([np.sum(dR_faceNorm[faces, :], axis = 0) for faces in model.vertex2face])

    for v in range(0, dR_a.shape[2]):
        dR_vNorm[:, :, v] = normalize(dR_vNorm[:, :, v])

    return dR_vNorm