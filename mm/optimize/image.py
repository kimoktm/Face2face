#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from scipy.interpolate import interp2d
from ..utils.mesh import calcNormals, generateFace, generateTexture, barycentricReconstruction
from ..utils.transform import rotMat2angle, sh9
from .derivative import dR_dpsi, dR_dtheta, dR_dphi, dR_normal, dR_sh

# delete
import autograd.numpy as np
import matplotlib.pyplot as plt


## CORRECT
def initialShapeCost(param, target, model,  w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.concatenate((param[model.numId + model.numExp:][3: 5], np.array([0])))
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)

    # After rigid transformation and scaling
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    rlan = (source - target.T).flatten('F')
    Elan = np.dot(rlan, rlan) / model.sourceLMInd.size

    # Regularization cost
    Ereg = np.sum(idCoef**2 / model.idEval) + np.sum(expCoef**2 / model.expEval)

    return (w[0] * Elan + w[1] * Ereg)

def initialShapeGrad(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]
    
    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)

    # After rigid transformation and scaling
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    rlan = (source - target.T).flatten('F')

    drV_dalpha = s * np.tensordot(R, model.idEvec[:, model.sourceLMInd, :], axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    return 2 * (w[0] * np.dot(Jlan.T, rlan) / model.sourceLMInd.size + w[1] * np.r_[idCoef / model.idEval, expCoef / model.expEval, np.zeros(6)])

def initialShapeResiuals(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.concatenate((param[model.numId + model.numExp:][3: 5], np.array([0])))
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)

    # After rigid transformation and scaling
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]


    w0 = (w[0] / model.sourceLMInd.size)**(1/2)
    w1 = w[1]**(1/2)

    # Reg cost not correct
    return np.concatenate((w0 * (source - target.T).flatten('F'), w1 * idCoef ** 2 / model.idEval, w1 * expCoef ** 2 / model.expEval))

def initialShapeJacobians(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]
    
    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)

    # After rigid transformation and scaling
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    drV_dalpha = s * np.tensordot(R, model.idEvec[:, model.sourceLMInd, :], axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    # Reg cost not correct
    eq2 = np.zeros((idCoef.size, param.size))
    eq2[:, :idCoef.size] = np.diag(idCoef / model.idEval)

    eq3 = np.zeros((expCoef.size, param.size))
    eq3[:, idCoef.size : idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / model.sourceLMInd.size)**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * Jlan, w1 * eq2, w1 * eq3]


def expResiuals(param, idCoef, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    param = np.r_[param[:-1], 0, param[-1]]

    # Landmark fitting cost
    source = generateFace(param, model, ind = model.sourceLMInd)[:2, :]

    w0 = (w[0] / model.sourceLMInd.size)**(1/2)
    w1 = w[1]**(1/2)

    # Reg cost not correct
    return np.r_[w0 * (source - target.T).flatten('F'), w1 * expCoef ** 2 / model.expEval]

def expJacobians(param, idCoef, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Rotation Euler angles, translation vector, scaling factor
    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]
    
    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)

    # After rigid transformation and scaling
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan = np.c_[drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId))
    eq2[:, :expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / model.sourceLMInd.size)**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * Jlan, w1 * eq2]



## CORRECT
def textureCost(texCoef, img, vertexCoord, model, renderObj, w = (1, 1), randomFaces = None):
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, model.texMean.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    # rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    
    # CPU rendering
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    reconstruction = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    # rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    # Color matching cost
    r = (reconstruction - img).flatten()
    Ecol = np.dot(r, r) / (numPixels * 3)

    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / model.texEval)

    return w[0] * Ecol + w[1] * Ereg

def textureGrad(texCoef, img, vertexCoord, model, renderObj, w = (1, 1), randomFaces = None):
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    numPixels = pixelFaces.size
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    pixelVertices = model.face[pixelFaces, :]
    
    r = (rendering - img).flatten('F')
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    return 2 * (w[0] * r.dot(J_texCoef) / (numPixels * 3) + w[1] * texCoef / model.texEval)

def textureResiduals(texCoef, img, vertexCoord, model, renderObj, w = (1, 1), randomFaces = None):
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w1 * texCoef ** 2 / model.texEval]

def textureJacobian(texCoef, img, vertexCoord, model, renderObj, w = (1, 1), randomFaces = None):
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[2:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    pixelVertices = model.face[pixelFaces, :]

    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        J_texCoef[c * numPixels: (c + 1) * numPixels, :] = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * J_texCoef, w1 * np.diag(texCoef / model.texEval)]



def textureLightingResiduals(texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    texCoef = texParam[:model.numTex]
    shCoef = texParam[model.numTex:].reshape(9, 3)

    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w1 * texCoef ** 2 / model.texEval]

def textureLightingJacobian(texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), randomFaces = None):
    texCoef = texParam[:model.numTex]
    shCoef = texParam[model.numTex:].reshape(9, 3)

    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[2:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    pixelVertices = model.face[pixelFaces, :]

    # pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    # pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    # J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)

    # xxx = np.zeros((pixelVertices.size, 27))
    # for c in range(3):
    #     xxx[c*numPixels: (c+1)*numPixels, c * 9 : (c+1) * 9] = barycentricReconstruction(sh * vertexColor[c, :], pixelFaces, pixelBarycentricCoords, model.face)

    xxx = np.zeros((pixelVertices.size, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            xxx[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]


    # print(np.array_equal(block_diag(*J_shCoef).shape, xxx))
    # print("##########")

    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)

    texCoefSide = np.r_[w0 * J_texCoef, w1 * np.diag(texCoef / model.texEval)]
    # shCoefSide = np.r_[w0 * block_diag(*J_shCoef), np.zeros((texCoef.size, shCoef.size))]
    shCoefSide = np.r_[w0 * xxx, np.zeros((texCoef.size, shCoef.size))]

    # print(J_texCoef.shape)
    # print(block_diag(*J_shCoef).shape)
    # print(texCoefSide.shape)
    # print(shCoefSide.shape)
    # print( np.c_[texCoefSide, shCoefSide].shape)

    return np.c_[texCoefSide, shCoefSide]



## CORRECT
def lightingCost(texParam, texCoef, img, vertexCoord, sh, model, renderObj, randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    texture = generateTexture(vertexCoord, np.concatenate((texCoef, texParam)), model)

    renderObj.updateVertexBuffer(np.concatenate((vertexCoord.T, model.texMean.T)))
    renderObj.resetFramebufferObject()
    renderObj.render()

    # CPU rendering
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    reconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, model.face)

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]

    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    r = (reconstruction - img).flatten()
    Ecol = np.dot(r, r) / (numPixels * 3)

    return Ecol

def lightingGrad(texParam, texCoef, img, vertexCoord, sh, model, renderObj, randomFaces = None):
    shCoef = texParam.reshape(9, 3)

    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    pixelVertices = model.face[pixelFaces, :]

    J_shCoef = np.zeros((pixelVertices.size, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            J_shCoef[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]

    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    rCol = (rendering.T - img.T).flatten()

    return 2 / (numPixels * 3) * rCol.dot(J_shCoef)

def lightingResiduals(texParam, texCoef, img, vertexCoord, sh, model, renderObj, randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    # shCoef = texParam.reshape(9, 3)

    texture = generateTexture(vertexCoord, np.concatenate((texCoef, texParam)), model)

    renderObj.updateVertexBuffer(np.concatenate((vertexCoord.T, model.texMean.T)))
    renderObj.resetFramebufferObject()
    renderObj.render()
    # rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]

    # CPU rendering
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    reconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, model.face)
    # reconstruction = np.zeros(rendering.shape)
    # reconstruction[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]

    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    # r = (reconstruction - img).flatten()
    # Ecol = np.dot(r, r) / (numPixels * 3)

    # return Ecol
    return (reconstruction - img).flatten('F')

def lightingJacobian(texParam, texCoef, img, vertexCoord, sh, model, renderObj, randomFaces = None):
    shCoef = texParam.reshape(9, 3)

    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[2:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    pixelVertices = model.face[pixelFaces, :]

    # pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    # pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    # J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    # J_shCoef = block_diag(*J_shCoef)

    xxx = np.zeros((pixelVertices.size, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            xxx[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]

    return xxx


## ~CORRECT
def bilinear_interpolate(im, y, x):
    # x = np.asarray(x)
    # y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    waI = np.multiply(wa[:, np.newaxis], Ia)
    wbI = np.multiply(wb[:, np.newaxis], Ib)
    wcI = np.multiply(wc[:, np.newaxis], Ic)
    wdI = np.multiply(wd[:, np.newaxis], Id)

    return waI + wbI + wcI + wdI

def denseCost(param, img, texCoef, model, renderObj, vertexCoord_1, w = (1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    param = np.concatenate((param[:-1], np.array([0, param[-1]])))

    # Generate face shape
    vertexCoord = generateFace(param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord_1.T, model.texMean.T])
    renderObj.resetFramebufferObject()
    renderObj.render()

    # CPU rendering
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    reconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, model.face)

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    w0 = (w[0] / (numPixels * 3))
    w1 = w[1]

    # Color matching cost
    r = (reconstruction - img).flatten()
    Ecol = np.dot(r, r) / (numPixels * 3)

    # Statistical regularization
    Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)

    return w[0] * Ecol + w[1] * Ereg

def denseCostV(param, img, texCoef, model, renderObj, vertexCoord_1, w = (1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    param = np.concatenate((param[:-1], np.array([0, param[-1]])))

    # Generate face shape
    vertexCoord = generateFace(param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord_1.T, model.texMean.T])
    renderObj.resetFramebufferObject()
    renderObj.render()

    # CPU rendering
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

    pixelVertices = np.unique(model.face[pixelFaces, :])
    visibleVertices = vertexCoord[:, pixelVertices]
    visibleColors = texture[:, pixelVertices]

    # print(visibleVertices.shape)
    # print(img[133, 165, :])
    # print(bilinear_interpolate(img, visibleVertices[0, 0:2], visibleVertices[1, 0:2]))
    # print(bilinear_interpolate(img, visibleVertices[0, :], visibleVertices[1, :]).shape)

    numVisibleVertices = visibleVertices.shape[1]
    diff = visibleColors.T - bilinear_interpolate(img, visibleVertices[0, :], visibleVertices[1, :])

    # Color matching cost
    r = diff.flatten()
    Ecol = np.dot(r, r) / (numVisibleVertices * 3)

    # Statistical regularization
    Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)

    return w[0] * Ecol + w[1] * Ereg


def denseGrad(param, img, texCoef, model, renderObj, w = (1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.empty((3, model.numVertices, 2))
    drV_dt[0, :] = [1, 0]
    drV_dt[1, :] = [0, 1]
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan = np.empty((2, model.numVertices, param.size))
    for l in range(2):
        Jlan[l] = np.c_[drV_dalpha[l, :], drV_ddelta[l, :],\
         drV_dpsi[l, :].flatten('F'), drV_dtheta[l, :].flatten('F'), drV_dphi[l, :].flatten('F'), drV_dt[l, :], drV_ds[l, :].flatten('F')]

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    for c in range(3):
        J_denseCoef[c * numPixels: (c + 1) * numPixels, :] = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])


    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    r = (rendering - img).flatten('F')

    reg = np.r_[idCoef / model.idEval, expCoef / model.expEval, np.zeros((6))]

    return 2 * (w[0] * r.dot(-J_denseCoef) / (numPixels * 3) + w[1] * reg)

def denseResiduals(param, img, texCoef, model, renderObj, w = (1, 1), randomFaces = None):
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    param = np.r_[param[:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w1 * idCoef ** 2 / model.idEval, w1 * expCoef ** 2 / model.expEval]

def denseJacobian(param, img, texCoef, model, renderObj, w = (1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # # shape derivatives
    # drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    # drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    # drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    # drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    # drV_dphi = s * np.dot(dR_dphi(angles), shape)
    # drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    # drV_ds = np.dot(R, shape)

    # # shape derivates in X, Y coordinates
    # Jlan_tmp = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
    #  drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    # Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size))

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    # drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_dt = np.empty((2, model.numVertices, 2))
    drV_dt[0, :] = [1, 0]
    drV_dt[1, :] = [0, 1]
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan = np.empty((2, model.numVertices, param.size))
    for l in range(2):
        Jlan[l] = np.c_[drV_dalpha[l, ...], drV_ddelta[l, ...],\
         drV_dpsi[l, :].flatten('F'), drV_dtheta[l, :].flatten('F'), drV_dphi[l, :].flatten('F'), drV_dt[l, :], drV_ds[l, :].flatten('F')]

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    for c in range(3):
        J_denseCoef[c * numPixels: (c + 1) * numPixels, :] = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])

    # set Identity to zeros
    #
    # for c in range(3):
    #     J_denseCoef[c * numPixels: (c + 1) * numPixels, : model.numId + model.numExp] = np.zeros((model.numId))

    # Reg cost not correct
    eq2 = np.zeros((idCoef.size, param.size))
    eq2[:, :idCoef.size] = np.diag(idCoef / model.idEval)

    eq3 = np.zeros((expCoef.size, param.size))
    eq3[:, idCoef.size : idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * -J_denseCoef, w1 * eq2, w1 * eq3]



def denseTexResiduals(param, img, model, renderObj, w = (1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex] 
    idCoef = param[model.numTex : model.numTex + model.numId]
    expCoef = param[model.numTex + model.numId: model.numTex + model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[model.numTex:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)
    w2 = w[2]**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w1 * texCoef ** 2 / model.texEval, w2 * idCoef ** 2 / model.idEval, w2 * expCoef ** 2 / model.expEval]

def denseTexJacobian(param, img, model, renderObj, w = (1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex] 
    idCoef = param[model.numTex : model.numTex + model.numId]
    expCoef = param[model.numTex + model.numId: model.numTex + model.numId + model.numExp]

    angles = param[model.numTex + model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numTex + model.numId + model.numExp:][3: 5], 0]
    s = param[model.numTex + model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan_tmp = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size - model.numTex))
    
    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numTex))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    for c in range(3):
        J_denseCoef[c * numPixels: (c + 1) * numPixels, :] = np.c_[barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face), -np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) - np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])]

    # Reg cost not correct
    eq2 = np.zeros((texCoef.size, param.size))
    eq2[:, :texCoef.size] = np.diag(texCoef / model.texEval)

    eq3 = np.zeros((idCoef.size, param.size))
    eq3[:, texCoef.size : texCoef.size + idCoef.size] = np.diag(idCoef / model.idEval)

    eq4 = np.zeros((expCoef.size, param.size))
    eq4[:, texCoef.size + idCoef.size : texCoef.size + idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / (numPixels * 3))**(1/2)
    w1 = w[1]**(1/2)
    w2 = w[2]**(1/2)

    return np.r_[w0 * J_denseCoef, w1 * eq2, w2 * eq3, w2 * eq4]


def denseShResiduals(param, img, shCoef, target, model, renderObj, w = (1, 1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex] 
    idCoef = param[model.numTex : model.numTex + model.numId]
    expCoef = param[model.numTex + model.numId: model.numTex + model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[model.numTex:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # landmakrs error
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    return np.r_[wcol * (rendering - img).flatten('F'), wlan * (source - target.T).flatten('F'), wreg_color * texCoef ** 2 / model.texEval, wreg_shape * idCoef ** 2 / model.idEval, wreg_shape * expCoef ** 2 / model.expEval]

def denseShJacobian(param, img, shCoef, target, model, renderObj, w = (1, 1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex] 
    idCoef = param[model.numTex : model.numTex + model.numId]
    expCoef = param[model.numTex + model.numId: model.numTex + model.numId + model.numExp]

    angles = param[model.numTex + model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numTex + model.numId + model.numExp:][3: 5], 0]
    s = param[model.numTex + model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan_tmp = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size - model.numTex))
    
    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numTex))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    # final dense jacobian (img_der * shape_der)
    vertexNorms = calcNormals(vertexCoord, model)
    # Evaluate spherical harmonics at face shape normals. The result is a (numVertices, 9) array where each column is a spherical harmonic basis for the 3DMM.
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
    shCoef = shCoef.reshape(9, 3)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        colr = pixelSHLighting * pixelTexEvecsCombo

        # colr = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        J_denseCoef[c * numPixels: (c + 1) * numPixels, :] = np.c_[colr, -np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) - np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])]


    # landmarks error
    shape_param = np.r_[param[model.numTex:-1], 0, param[-1]]
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    drV_dalpha = s * np.tensordot(R, model.idEvec[:, model.sourceLMInd, :], axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan_landmarks = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    Jlan_landmarks = np.c_[np.zeros((target.size, model.numTex)), Jlan_landmarks]


    # weighting
    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # Reg cost not correct
    eq2 = np.zeros((texCoef.size, param.size))
    eq2[:, :texCoef.size] = np.diag(texCoef / model.texEval)

    eq3 = np.zeros((idCoef.size, param.size))
    eq3[:, texCoef.size : texCoef.size + idCoef.size] = np.diag(idCoef / model.idEval)

    eq4 = np.zeros((expCoef.size, param.size))
    eq4[:, texCoef.size + idCoef.size : texCoef.size + idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    return np.r_[wcol * J_denseCoef, wlan * Jlan_landmarks, wreg_color * eq2, wreg_shape * eq3, wreg_shape * eq4]



def denseAllResiduals(param, img, model, renderObj, w = (1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex]
    shCoef = param[model.numTex : model.numTex + 27].reshape(9, 3)
    idCoef = param[model.numTex + 27 : model.numTex + 27 + model.numId]
    expCoef = param[model.numTex + 27 + model.numId: model.numTex + 27 + model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[model.numTex + 27 :-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces = renderObj.grabRendering(return_info = True)[:3]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wreg_color = w[1]**(1/2)
    wreg_shape = w[2]**(1/2)

    return np.r_[wcol * (rendering - img).flatten('F'), wreg_color * texCoef ** 2 / model.texEval, wreg_shape * idCoef ** 2 / model.idEval, wreg_shape * expCoef ** 2 / model.expEval]

def denseAllJacobian(param, img, model, renderObj, w = (1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex]
    shCoef = param[model.numTex : model.numTex + 27].reshape(9, 3)
    idCoef = param[model.numTex + 27 : model.numTex + 27 + model.numId]
    expCoef = param[model.numTex + 27 + model.numId: model.numTex + 27 + model.numId + model.numExp]

    angles = param[model.numTex + 27 + model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numTex + 27 + model.numId + model.numExp:][3: 5], 0]
    s = param[model.numTex + 27 + model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan_tmp = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size - model.numTex - 27))

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numTex - 27))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    #
    # Derivatives
    #
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    # Albedo derivative
    J_texCoef = np.empty((numPixels * 3, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    # Sh derivative
    J_shCoef = np.zeros((numPixels * 3, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            J_shCoef[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]


    # Shape derivative
    drV_dt = np.tile(np.eye(3), [model.numVertices, 1])
    Klan_tmp = np.c_[drV_dalpha[:3, ...].reshape((vertexCoord.size, idCoef.size), order = 'F'), drV_ddelta[:3, ...].reshape((vertexCoord.size, expCoef.size), order = 'F'),\
     drV_dpsi[:3, :].flatten('F'), drV_dtheta[:3, :].flatten('F'), drV_dphi[:3, :].flatten('F'), drV_dt[:,:2], drV_ds[:3, :].flatten('F')]
    Klan = np.reshape(Klan_tmp, (3, model.numVertices, param.size - model.numTex - 27))
    xxx = dR_normal(vertexCoord, model, Klan)
    zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numTex - 27))
    for c in range(3):
        lll = np.empty((zzz.shape[1:]))
        for v in range(0, zzz.shape[2]):
            lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] =  shLighting - imgDer

    # # remove shape derivatices
    # J_mask = np.zeros((numPixels * 3, param.size - model.numTex - 27))
    # for c in range(3):
    #     J_mask[c * numPixels : (c + 1) * numPixels, -6: ] = 1.0
    # J_denshapeCoef  = J_denshapeCoef * J_mask

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    J_denseCoef = np.c_[J_texCoef, J_shCoef, J_denshapeCoef]


    # Reg cost not correct
    eq2 = np.zeros((texCoef.size, param.size))
    eq2[:, :texCoef.size] = np.diag(texCoef / model.texEval)

    eq3 = np.zeros((idCoef.size, param.size))
    eq3[:, texCoef.size + 27 : texCoef.size + 27 + idCoef.size] = np.diag(idCoef / model.idEval)

    eq4 = np.zeros((expCoef.size, param.size))
    eq4[:, texCoef.size + 27 + idCoef.size : texCoef.size + 27 + idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wreg_color = w[1]**(1/2)
    wreg_shape = w[2]**(1/2)

    return np.r_[wcol * J_denseCoef, wreg_color * eq2, wreg_shape * eq3, wreg_shape * eq4]


def denseExpResiduals(param, idCoef, texCoef, shCoef, target, img, model, renderObj, w = (1, 1, 1), randomFaces = None):
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    param = np.r_[param[:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg = w[2]**(1/2)

    # landmakrs error
    source = generateFace(param, model, ind = model.sourceLMInd)[:2, :]
    
    return np.r_[wcol * (rendering - img).flatten('F'), wlan * (source - target.T).flatten('F'), wreg * expCoef ** 2 / model.expEval]

def denseExpJacobian(param, idCoef, texCoef, shCoef, target, img, model, renderObj, w = (1, 1, 1), randomFaces = None):
    # Shape eigenvector coefficients
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan_tmp = np.c_[drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size - model.numId))
    
    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numId))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size - model.numId))
    for c in range(3):
        J_denseCoef[c * numPixels: (c + 1) * numPixels, :] = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])


    # landmarks error
    shape_param = np.r_[param[:-1], 0, param[-1]]
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan_landmarks = np.c_[drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    # weighting
    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg = w[2]**(1/2)

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId))
    eq2[:, :expCoef.size] = np.diag(expCoef / model.expEval)

    return np.r_[wcol * -J_denseCoef, wlan * Jlan_landmarks, wreg * eq2]



## ~CORRECT - Auto diff ignores SH influences on shape??
def denseJointCost(param, img, target, model, renderObj, w = (1, 1, 1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex]
    shCoef = param[model.numTex : model.numTex + 27].reshape(9, 3)
    idCoef = param[model.numTex + 27 : model.numTex + 27 + model.numId]
    expCoef = param[model.numTex + 27 + model.numId: model.numTex + 27 + model.numId + model.numExp]

    # Insert z translation
    shape_param = np.concatenate((param[model.numTex + 27 :-1], np.array([0, param[-1]])))

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, model.texMean.T])
    renderObj.resetFramebufferObject()
    renderObj.render()

    # CPU rendering
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    reconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, model.face)

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # landmakrs error
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    # Color matching cost
    r = (reconstruction - img).flatten()
    Ecol = np.dot(r, r) / (numPixels * 3)

    rlan = (source - target.T).flatten('F')
    Elan = np.dot(rlan, rlan) / model.sourceLMInd.size

    # Statistical regularization
    Creg = np.sum(texCoef ** 2 / model.texEval)
    Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)

    return w[0] * Ecol + w[1] * Elan + w[2] * Creg + w[3] * Ereg

def denseJointGrad(param, img, target, model, renderObj, w = (1, 1, 1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex]
    shCoef = param[model.numTex : model.numTex + 27].reshape(9, 3)
    idCoef = param[model.numTex + 27 : model.numTex + 27 + model.numId]
    expCoef = param[model.numTex + 27 + model.numId: model.numTex + 27 + model.numId + model.numExp]

    angles = param[model.numTex + 27 + model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numTex + 27 + model.numId + model.numExp:][3: 5], 0]
    s = param[model.numTex + 27 + model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelCoord = pixelCoord[randomFaces, :]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan_tmp = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size - model.numTex - 27))

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numTex - 27))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    #
    # Derivatives
    #
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    # Albedo derivative
    J_texCoef = np.empty((numPixels * 3, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c * numPixels: (c + 1) * numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    # Sh derivative
    # pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    # pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    # J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    # J_shCoef = block_diag(*J_shCoef)

    J_shCoef = np.zeros((numPixels * 3, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            J_shCoef[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]


    # Shape derivative
    drV_dt = np.tile(np.eye(3), [model.numVertices, 1])
    Klan_tmp = np.c_[drV_dalpha[:3, ...].reshape((vertexCoord.size, idCoef.size), order = 'F'), drV_ddelta[:3, ...].reshape((vertexCoord.size, expCoef.size), order = 'F'),\
     drV_dpsi[:3, :].flatten('F'), drV_dtheta[:3, :].flatten('F'), drV_dphi[:3, :].flatten('F'), drV_dt[:,:2], drV_ds[:3, :].flatten('F')]
    Klan = np.reshape(Klan_tmp, (3, model.numVertices, param.size - model.numTex - 27))
    xxx = dR_normal(vertexCoord, model, Klan)
    zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numTex - 27))
    for c in range(3):
        lll = np.empty((zzz.shape[1:]))
        for v in range(0, zzz.shape[2]):
            lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] = shLighting - imgDer

    # print(J_denshapeCoef)
    # print(J_texCoef.shape)
    # print(J_shCoef.shape)
    # print(J_denshapeCoef.shape)

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    J_denseCoef = np.c_[J_texCoef, J_shCoef, J_denshapeCoef]
    # print(J_denseCoef.shape)
    # print("######################")

    # landmarks error
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    drV_dalpha = s * np.tensordot(R, model.idEvec[:, model.sourceLMInd, :], axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan_landmarks = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    # Jlan_landmarks = np.c_[np.zeros((target.size, model.numTex + 27)), Jlan_landmarks]

    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    r = (rendering - img).flatten('F')

    rlan = (source - target.T).flatten('F')

    reg = np.r_[w[2] * texCoef / model.texEval, np.zeros((27)), w[3] * idCoef / model.idEval, w[3] * expCoef / model.expEval, np.zeros((6))]

    return 2 * (w[0] * r.dot(J_denseCoef) / (numPixels * 3) + w[1] * np.r_[np.zeros(model.numTex + 27), np.dot(Jlan_landmarks.T, rlan)] / model.sourceLMInd.size + reg)

def denseJointResiduals(param, img, target, model, renderObj, w = (1, 1, 1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex]
    shCoef = param[model.numTex : model.numTex + 27].reshape(9, 3)
    idCoef = param[model.numTex + 27 : model.numTex + 27 + model.numId]
    expCoef = param[model.numTex + 27 + model.numId: model.numTex + 27 + model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[model.numTex + 27 :-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces = renderObj.grabRendering(return_info = True)[:3]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # landmakrs error
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    return np.r_[wcol * (rendering - img).flatten('F'), wlan * (source - target.T).flatten('F'), wreg_color * texCoef ** 2 / model.texEval, wreg_shape * idCoef ** 2 / model.idEval, wreg_shape * expCoef ** 2 / model.expEval]

def denseJointJacobian(param, img, target, model, renderObj, w = (1, 1, 1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    texCoef = param[: model.numTex]
    shCoef = param[model.numTex : model.numTex + 27].reshape(9, 3)
    idCoef = param[model.numTex + 27 : model.numTex + 27 + model.numId]
    expCoef = param[model.numTex + 27 + model.numId: model.numTex + 27 + model.numId + model.numExp]

    angles = param[model.numTex + 27 + model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numTex + 27 + model.numId + model.numExp:][3: 5], 0]
    s = param[model.numTex + 27 + model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelCoord = pixelCoord[randomFaces, :]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.empty((3, model.numVertices, 2))
    drV_dt[0, :] = [1, 0]
    drV_dt[1, :] = [0, 1]
    drV_dt[2, :] = [0, 0]
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan = np.empty((3, model.numVertices, param.size - model.numTex - 27))
    for l in range(3):
        Jlan[l] = np.c_[drV_dalpha[l, :], drV_ddelta[l, :],\
         drV_dpsi[l, :].flatten('F'), drV_dtheta[l, :].flatten('F'), drV_dphi[l, :].flatten('F'), drV_dt[l, :], drV_ds[l, :].flatten('F')]

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numTex - 27))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    #
    # Derivatives
    #
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    # Albedo derivative
    J_texCoef = np.empty((numPixels * 3, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c * numPixels: (c + 1) * numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    # Sh derivative
    # pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    # pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    # J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    # J_shCoef = block_diag(*J_shCoef)

    J_shCoef = np.zeros((numPixels * 3, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            J_shCoef[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]

    # Shape derivative
    # xxx = dR_normal(vertexCoord, model, Jlan)
    # zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numTex - 27))
    for c in range(3):
        # lll = np.empty((zzz.shape[1:]))
        # for v in range(0, zzz.shape[2]):
        #     lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        # shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] = - imgDer

    # print(J_texCoef.shape)
    # print(J_shCoef.shape)
    # print(J_denshapeCoef.shape)

    # final dense jacobian (img_der * shape_der)
    # J_denseCoef = np.empty((numPixels * 3, param.size))
    # print(J_denseCoef.shape)
    # print("######################")

    # landmarks error
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    drV_dalpha = s * np.tensordot(R, model.idEvec[:, model.sourceLMInd, :], axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan_landmarks = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    Jlan_landmarks = np.c_[np.zeros((target.size, model.numTex + 27)), Jlan_landmarks]


    # weighting
    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # Reg cost not correct
    eq2 = np.zeros((texCoef.size, param.size))
    eq2[:, :texCoef.size] = np.diag(texCoef / model.texEval)

    eq3 = np.zeros((idCoef.size, param.size))
    eq3[:, texCoef.size + 27 : texCoef.size + 27 + idCoef.size] = np.diag(idCoef / model.idEval)

    eq4 = np.zeros((expCoef.size, param.size))
    eq4[:, texCoef.size + 27 + idCoef.size : texCoef.size + 27 + idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    J_denseCoef = np.c_[J_texCoef, J_shCoef, J_denshapeCoef]

    return np.r_[wcol * J_denseCoef, wlan * Jlan_landmarks, wreg_color * eq2, wreg_shape * eq3, wreg_shape * eq4]



def denseJointExpResiduals(param, idCoef, texCoef, img, target, model, renderObj, w = (1, 1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    shCoef = param[: 27].reshape(9, 3)
    param = np.r_[idCoef, param[27:]]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces = renderObj.grabRendering(return_info = True)[:3]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]] * 1.3
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_shape = w[2]**(1/2)

    # landmakrs error
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    return np.r_[wcol * (rendering - img).flatten('F'), wlan * (source - target.T).flatten('F'), wreg_shape * expCoef ** 2 / model.expEval]

def denseJointExpJacobian(param, idCoef, texCoef, img, target, model, renderObj, w = (1, 1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    shCoef = param[: 27].reshape(9, 3)
    param = np.r_[idCoef, param[27:]]
    expCoef = param[model.numId: model.numId + model.numExp]

    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelCoord = pixelCoord[randomFaces, :]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.empty((3, model.numVertices, 2))
    drV_dt[0, :] = [1, 0]
    drV_dt[1, :] = [0, 1]
    drV_dt[2, :] = [0, 0]
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan = np.empty((3, model.numVertices, param.size - model.numId))
    for l in range(3):
        Jlan[l] = np.c_[drV_ddelta[l, ...],\
         drV_dpsi[l, :].flatten('F'), drV_dtheta[l, :].flatten('F'), drV_dphi[l, :].flatten('F'), drV_dt[l, :], drV_ds[l, :].flatten('F')]

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numId))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    #
    # Derivatives
    #
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    # Sh derivative
    J_shCoef = np.zeros((numPixels * 3, 27))
    for c in range(3):
        val = barycentricReconstruction(vertexColor[c, :] * sh, pixelFaces, pixelBarycentricCoords, model.face)
        for i in range(9):
            J_shCoef[c*numPixels: (c+1)*numPixels, c + i * 3] = val[:, i]

    # Shape derivative
    # xxx = dR_normal(vertexCoord, model, Jlan)
    # zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numId))
    for c in range(3):
        # lll = np.empty((zzz.shape[1:]))
        # for v in range(0, zzz.shape[2]):
        #     lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        # shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] =  -imgDer

    # final dense jacobian (img_der * shape_der)
    # J_denseCoef = np.empty((numPixels * 3, param.size - model.numId + 27))
    # J_denseCoef = np.c_[J_shCoef, J_denshapeCoef]
    # print(J_denseCoef.shape)
    # print("######################")

    # landmarks error
    shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, model.sourceLMInd, :], expCoef, axes = 1)
    source = (s * np.dot(R, shape) + t[:, np.newaxis])[:2, :]

    drV_ddelta = s * np.tensordot(R, model.expEvec[:, model.sourceLMInd, :], axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.sourceLMInd.size, 1])
    drV_ds = np.dot(R, shape)

    Jlan_landmarks = np.c_[drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]

    Jlan_landmarks = np.c_[np.zeros((target.size, 27)), Jlan_landmarks]

    # weighting
    wcol = (w[0] / (numPixels * 3))**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_shape = w[2]**(1/2)

    J_denseCoef = np.c_[J_shCoef, J_denshapeCoef]

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId + 27))
    eq2[:, 27 : 27 + expCoef.size] = np.diag(expCoef / model.expEval)

    return np.r_[wcol * J_denseCoef, wlan * Jlan_landmarks, wreg_shape * eq2]



## Dense expression only
def denseExpOnlyCost(param, idCoef, texCoef, shCoef, img, model, renderObj, w = (1, 1), randomFacesNum = None):
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)
    # texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces = renderObj.grabRendering(return_info = True)[:3]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]

    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]] * 1.0
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]] * 1.0

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wreg_shape = w[1]**(1/2)

    # Color matching cost
    r = (rendering - img).flatten()
    Ecol = np.dot(r, r) / (numPixels * 3)

    # Statistical regularization
    Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)

    return w[0] * Ecol + w[1] * Ereg

def denseExpOnlyGrad(param, idCoef, texCoef, shCoef, img, model, renderObj, w = (1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)
    # texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelCoord = pixelCoord[randomFaces, :]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.tile(np.eye(2), [model.numVertices, 1])
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan_tmp = np.c_[ drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'),\
     drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size - model.numId))

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numId))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    #
    # Derivatives
    #
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    # Shape derivative
    # drV_dt = np.tile(np.eye(3), [model.numVertices, 1])
    # Klan_tmp = np.c_[drV_ddelta[:3, ...].reshape((vertexCoord.size, expCoef.size), order = 'F'),\
    #  drV_dpsi[:3, :].flatten('F'), drV_dtheta[:3, :].flatten('F'), drV_dphi[:3, :].flatten('F'), drV_dt[:,:2], drV_ds[:3, :].flatten('F')]
    # Klan = np.reshape(Klan_tmp, (3, model.numVertices, param.size - model.numId))
    # xxx = dR_normal(vertexCoord, model, Klan)
    # zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.zeros((numPixels * 3, param.size - model.numId))
    for c in range(3):
        # lll = np.empty((zzz.shape[1:]))
        # for v in range(0, zzz.shape[2]):
        #     lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        # shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        # imgDer = (J_shapeCoef[0, :, :]) + (J_shapeCoef[1, :, :])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] = -imgDer 
    J_denseCoef = J_denshapeCoef


    # Reg cost not correct
    # np.savetxt("/home/karim/Desktop/jacob.csv", J_denseCoef * 100000, delimiter = ',')

    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    r = (rendering - img).flatten('F')

    reg = np.r_[w[1] * expCoef / model.expEval, np.zeros((6))]

    return 2 * (w[0] * r.dot(J_denseCoef) / (numPixels * 3) + reg)


def denseExpOnlyResiduals(param, idCoef, texCoef, shCoef, img, model, renderObj, w = (1, 1), randomFacesNum = None):
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    shape_param = np.r_[param[:-1], 0, param[-1]]

    # Generate face shape
    vertexCoord = generateFace(shape_param, model)

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces = renderObj.grabRendering(return_info = True)[:3]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]

    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]] * 1.0
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]] * 1.0

    wcol = (w[0] / (numPixels * 3))**(1/2)
    wreg_shape = w[1]**(1/2)

    return np.r_[wcol * (rendering - img).flatten('F'),  wreg_shape * expCoef ** 2 / model.expEval]

def denseExpOnlyJacobian(param, idCoef, texCoef, shCoef, img, model, renderObj, w = (1, 1), randomFacesNum = None):
    # Shape eigenvector coefficients
    param = np.r_[idCoef, param]
    expCoef = param[model.numId: model.numId + model.numExp]

    angles = param[model.numId + model.numExp:][:3]
    R = rotMat2angle(angles)
    t = np.r_[param[model.numId + model.numExp:][3: 5], 0]
    s = param[model.numId + model.numExp:][5]

    # The eigenmodel, before rigid transformation and scaling
    shape = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)

    vertexCoord = s * np.dot(R, shape) + t[:, np.newaxis]

    # After rigid transformation and scaling
    source = vertexCoord[:2, :]

    # Generate the texture at the 3DMM vertices from the learned texture coefficients
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)

    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[1:]

    if randomFacesNum is not None:
        randomFaces = np.random.randint(0, pixelFaces.size, randomFacesNum)
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelCoord = pixelCoord[randomFaces, :]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size

    # shape derivatives
    # drV_dalpha = s * np.tensordot(R, model.idEvec, axes = 1)
    drV_ddelta = s * np.tensordot(R, model.expEvec, axes = 1)
    drV_dpsi = s * np.dot(dR_dpsi(angles), shape)
    drV_dtheta = s * np.dot(dR_dtheta(angles), shape)
    drV_dphi = s * np.dot(dR_dphi(angles), shape)
    drV_dt = np.empty((3, model.numVertices, 2))
    drV_dt[0, :] = [1, 0]
    drV_dt[1, :] = [0, 1]
    drV_dt[2, :] = [0, 0]
    drV_ds = np.dot(R, shape)

    # shape derivates in X, Y coordinates
    Jlan = np.empty((3, model.numVertices, param.size - model.numId))
    for l in range(3):
        Jlan[l] = np.c_[drV_ddelta[l, ...],\
         drV_dpsi[l, :].flatten('F'), drV_dtheta[l, :].flatten('F'), drV_dphi[l, :].flatten('F'), drV_dt[l, :], drV_ds[l, :].flatten('F')]

    # vertex space to pixel space
    J_shapeCoef = np.empty((2, numPixels, param.size - model.numId))
    for c in range(2):
        J_shapeCoef[c, :, :] = barycentricReconstruction(Jlan[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    # img derivative in each channel
    img_grad_x = np.empty(img.shape)
    img_grad_y = np.empty(img.shape)
    for c in range(3):
        x = np.asarray(np.gradient(np.array(img[:, :, c], dtype = float)))
        img_grad_y[:, :, c] = x[0, :, :]
        img_grad_x[:, :, c] = x[1, :, :]

    img_grad_x = img_grad_x[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_grad_y = img_grad_y[pixelCoord[:, 0], pixelCoord[:, 1]]

    #
    # Derivatives
    #
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    # Shape derivative
    # xxx = dR_normal(vertexCoord, model, Jlan)
    # zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.zeros((numPixels * 3, param.size - model.numId))
    for c in range(3):
        # lll = np.empty((zzz.shape[1:]))
        # for v in range(0, zzz.shape[2]):
        #     lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        # shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] = -imgDer

    J_denseCoef = J_denshapeCoef
    wcol = (w[0] / (numPixels * 3))**(1/2)
    wreg_shape = w[1]**(1/2)

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId))
    eq2[:, : expCoef.size] = np.diag(expCoef / model.expEval)

    return np.r_[wcol * J_denseCoef, wreg_shape * eq2]