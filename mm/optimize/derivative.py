#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains helper functions that take the derivative of certain functions in the rendering pipeline.
"""

import numpy as np

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