#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions to be used with the scipy.optimize package in order to fit the 3DMM to a target depth map.
"""

import numpy as np
from ..utils.mesh import generateFace
from ..utils.transform import rotMat2angle
from .derivative import dR_dpsi, dR_dtheta, dR_dphi

def initialShapeCost(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]
    
    # Landmark fitting cost
    source = generateFace(param, model, ind = model.sourceLMInd)
    
    rlan = (source - target.T).flatten('F')
    Elan = np.dot(rlan, rlan) / model.sourceLMInd.size
    
    # Regularization cost
    Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)
    
    return w[0] * Elan + w[1] * Ereg

def initialShapeGrad(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = param[model.numId + model.numExp:][3: 6]
    s = param[model.numId + model.numExp:][6]
    
    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)
    
    # After rigid transformation and scaling
    source = s*np.dot(R, shape) + t[:, np.newaxis]
    
    rlan = (source - target.T).flatten('F')
        
    drV_dalpha = s*np.tensordot(R, model.idEvec[:, model.sourceLMInd, :], axes = 1)
    drV_ddelta = s*np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s*np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s*np.dot(dR_dtheta(angles), shape)
    drV_dphi = s*np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(3), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)
    
    Jlan = np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
    
    return 2 * (w[0] * np.dot(Jlan.T, rlan) / model.sourceLMInd.size + w[1] * np.r_[idCoef / model.idEval, expCoef / model.expEval, np.zeros(7)])

def shapeCost(param, model, target, targetLandmarks, NN, w = (1, 1, 1), calcID = True):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]
    
    # Transpose target if necessary
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # After rigid transformation and scaling
    source = generateFace(param, model)
    
    # Find the nearest neighbors of the target to the source vertices
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
    
    # Calculate resisduals
    rver = (source - targetNN).flatten('F')
    rlan = (source[:, model.sourceLMInd] - targetLandmarks).flatten('F')
    
    # Calculate costs
    Ever = np.dot(rver, rver) / model.numVertices
    Elan = np.dot(rlan, rlan) / model.sourceLMInd.size
    
    if calcID:
        Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)
    else:
        Ereg = np.sum(expCoef ** 2 / model.expEval)
    
    return w[0] * Ever + w[1] * Elan + w[2] * Ereg

def shapeGrad(param, model, target, targetLandmarks, NN, w = (1, 1, 1), calcID = True):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = param[model.numId + model.numExp:][3: 6]
    s = param[model.numId + model.numExp:][6]
    
    # Transpose if necessary
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)
    
    # After rigid transformation and scaling
    source = s*np.dot(R, shape) + t[:, np.newaxis]
    
    # Find the nearest neighbors of the target to the source vertices
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
    
    # Calculate resisduals
    rver = (source - targetNN).flatten('F')
    rlan = (source[:, model.sourceLMInd] - targetLandmarks).flatten('F')
        
    drV_ddelta = s*np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s*np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s*np.dot(dR_dtheta(angles), shape)
    drV_dphi = s*np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(3), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)
    
    if calcID:
        
        drV_dalpha = s*np.tensordot(R, model.idEvec, axes = 1)
        
        Jver = np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
        
        Jlan = np.c_[drV_dalpha[:, model.sourceLMInd, :].reshape((targetLandmarks.size, idCoef.size), order = 'F'), drV_ddelta[:, model.sourceLMInd, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, model.sourceLMInd].flatten('F'), drV_dtheta[:, model.sourceLMInd].flatten('F'), drV_dphi[:, model.sourceLMInd].flatten('F'), drV_dt[:model.sourceLMInd.size * 3, :], drV_ds[:, model.sourceLMInd].flatten('F')]
        
        return 2 * (w[0] * np.dot(Jver.T, rver) / model.numVertices + w[1] * np.dot(Jlan.T, rlan) / model.sourceLMInd.size + w[2] * np.r_[idCoef / model.idEval, expCoef / model.expEval, np.zeros(7)])
    
    else:
        
        Jver = np.c_[drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
        
        Jlan = np.c_[drV_ddelta[:, model.sourceLMInd, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, model.sourceLMInd].flatten('F'), drV_dtheta[:, model.sourceLMInd].flatten('F'), drV_dphi[:, model.sourceLMInd].flatten('F'), drV_dt[:model.sourceLMInd.size * 3, :], drV_ds[:, model.sourceLMInd].flatten('F')]
        
        return 2 * (np.r_[np.zeros(idCoef.size), w[0] * np.dot(Jver.T, rver) / model.numVertices] + np.r_[np.zeros(idCoef.size), w[1] * np.dot(Jlan.T, rlan) / model.sourceLMInd.size] + w[2] * np.r_[np.zeros(idCoef.size), expCoef / model.expEval, np.zeros(7)])