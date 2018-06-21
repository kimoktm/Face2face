#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions related to finding the optimal orthographic and perspective transforms and camera matrices to align the 3DMM with source depth maps or images. They use sparse landmark correspondences between the 3DMM and the source to find an optimal initialization.
"""

import numpy as np
from ..utils.transform import rotMat2angle
from scipy.linalg import rq
from scipy.optimize import least_squares

def initialRegistration(A, B):
    """Performs the Kabsch algorithm to find the optimal similarity transform parameters between 3D-3D landmark correspondences.
    
    Args:
        A (ndarray): Set of source vertices
        B (ndarray): Set of target vertices such at B' = s*R*A.T + t
    
    Returns:
        ndarray, (7,): The optimal Euler angles, the 3D translation vector, and the scaling factor
    """
    
    # Make sure the x, y, z vertex coordinates are along the columns
    if A.shape[0] == 3:
        A = A.T
    if B.shape[0] == 3:
        B = B.T
    
    # Find centroids of A and B landmarks and move them to the origin
    muA = np.mean(A, axis = 0)
    muB = np.mean(B, axis = 0)
    A = A - muA
    B = B - muB
    
    # Calculate the rotation matrix R. Note that the returned V is actually V.T.
    U, V = np.linalg.svd(np.dot(A.T, B))[::2]
    R = np.dot(V.T, U.T)
    
    # Flip sign on the third column of R if it is a reflectance matrix
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        
    # Find scale factor
    s = np.trace(np.dot(B.T, np.dot(A, R.T))) / np.trace(np.dot(A.T, A))
    
    # Find the translation vector
    t = -s*np.dot(R, muA) + muB
    
    # Find Euler angles underlying rotation matrix
    angles = rotMat2angle(R)
    
    return np.r_[angles, t, s]

def estimateCamMat(lm2D, lm3D, cam = 'orthographic'):
    """Estimates camera matrix from 2D-3D landmark correspondences using the Direct linear transform / "Gold Standard Algorithm".
    
    For an orthographic camera, the algebraic and geometric errors in the algorithm are equivalent, so there is no need to do the least squares step at the end. The orthographic camera returns a 2x4 camera matrix, since the third row is just [0, 0, 0, 1].
    
    Args:
        lm2D (ndarray, (n, 2)): landmark x, y coordinates in an image
        lm3D (ndarray, (n, 3)): landmark x, y, z coordinates in the 3DMM
        cam (str): 'perspective' or 'orthographic' to determine the type of camera projection matrix to return
    
    Returns:
        ndarray: Camera projection matrix, (2, 4) if 'orthrographic', (3, 4) if 'perspective'
    """
    # Normalize landmark coordinates; preconditioning
    numLandmarks = lm2D.shape[0]
    
    c2D = np.mean(lm2D, axis = 0)
    uvCentered = lm2D - c2D
    s2D = np.linalg.norm(uvCentered, axis = 1).mean()
    
    c3D = np.mean(lm3D, axis = 0)
    xyzCentered = lm3D - c3D
    s3D = np.linalg.norm(xyzCentered, axis = 1).mean()
    X = np.c_[xyzCentered / s3D * np.sqrt(3), np.ones(numLandmarks)]
    
    # Similarity transformations for normalization
    Tinv = np.array([[s2D, 0, c2D[0]], [0, s2D, c2D[1]], [0, 0, 1]])
    U = np.linalg.inv([[s3D, 0, 0, c3D[0]], [0, s3D, 0, c3D[1]], [0, 0, s3D, c3D[2]], [0, 0, 0, 1]])
    
    if cam == 'orthographic':
        x = uvCentered / s2D * np.sqrt(2)
        
        # Build linear system of equations in 8 unknowns of projection matrix
        A = np.zeros((2 * numLandmarks, 8))
        
        A[0: 2*numLandmarks - 1: 2, :4] = X
        A[1: 2*numLandmarks: 2, 4:] = X
        
        # Solve linear system and de-normalize
        p8 = np.linalg.lstsq(A, x.flatten())[0].reshape(2, 4)
        Pnorm = np.vstack((p8, np.array([0, 0, 0, 1])))
        P = Tinv.dot(Pnorm).dot(U)
        
        return P[:2, :]
    
    elif cam == 'perspective':
        x = np.c_[uvCentered / s2D * np.sqrt(2), np.ones(numLandmarks)]
        
        # Matrix for homogenous system of equations to solve for camera matrix
        A = np.zeros((2 * numLandmarks, 12))
        
        A[0: 2*numLandmarks - 1: 2, 0: 4] = X
        A[0: 2*numLandmarks - 1: 2, 8:] = -x[:, 0, np.newaxis] * X
        
        A[1: 2*numLandmarks: 2, 4: 8] = -X
        A[1: 2*numLandmarks: 2, 8:] = x[:, 1, np.newaxis] * X
        
        # Take the SVD and take the last row of V', which corresponds to the lowest eigenvalue, as the homogenous solution
        V = np.linalg.svd(A, full_matrices = 0)[-1]
        Pnorm = np.reshape(V[-1, :], (3, 4))
        
        # Further nonlinear LS to minimize error between 2D landmarks and 3D projections onto 2D plane.
        def cameraProjectionResidual(M, x, X):
            """
            min_{P} sum_{i} || x_i - PX_i ||^2
            """
            return x.flatten() - np.dot(X, M.reshape((3, 4)).T).flatten()
        
        Pgold = least_squares(cameraProjectionResidual, Pnorm.flatten(), args = (x, X))
        
        # Denormalize P
        P = Tinv.dot(Pgold.x.reshape(3, 4)).dot(U)
        
        return P

def splitCamMat(P, cam = 'orthographic'):
    """Splits the camera projection matrix into relevant intrinsic and extrinsic parameters.
    
    Args:
        P (ndarray): A camera projection matrix
        cam (str): 'perspective' or 'orthographic' to determine how to split the camera projection matrix
    
    Returns:
        (tuple): tuple containing:
            
            K (ndarray (1,) or (3, 3)): orthographic scale parameter or perspective intrinsic camera matrix
            angles (ndarray (3,)): Euler angles
            t (ndarray (2,)): translation vector
    """
    if cam == 'orthographic':
        # Extract params from orthographic projection matrix
        R1 = P[0, 0: 3]
        R2 = P[1, 0: 3]
        t = np.r_[P[0, 3], P[1, 3]]
        
        K = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
        r1 = R1 / np.linalg.norm(R1)
        r2 = R2 / np.linalg.norm(R2)
        r3 = np.cross(r1, r2)
        R = np.vstack((r1, r2, r3))
        
        # Set R to closest orthogonal matrix to estimated rotation matrix
        U, V = np.linalg.svd(R)[::2]
        R = U.dot(V)
        
        # Determinant of R must = 1
        if np.linalg.det(R) < 0:
            U[2, :] = -U[2, :]
            R = U.dot(V)
        
        # Remove scale from translations
#        t = t / K
        
        angles = rotMat2angle(R)
        
        return K, angles, t
    
    elif cam == 'perspective':
        # Get inner parameters from projection matrix via RQ decomposition
        K, R = rq(P[:, :3], mode = 'economic')
        angles = rotMat2angle(R)
        t = np.linalg.inv(K).dot(P[:, -1])
        
        return K, angles, t