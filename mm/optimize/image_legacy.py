#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from ..utils.mesh import generateFace, generateTexture, barycentricReconstruction
from ..utils.transform import rotMat2angle
from .derivative import dR_dpsi, dR_dtheta, dR_dphi

def initialShapeCost(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]
    
    # Insert z translation
    param = np.r_[param[:-1], 0, param[-1]]

    # Landmark fitting cost
    source = generateFace(param, model, ind = model.sourceLMInd)[:2, :]
    
    rlan = (source - target.T).flatten('F')
    Elan = np.dot(rlan, rlan) / model.sourceLMInd.size

    # Regularization cost
    Ereg = np.sum(idCoef**2 / model.idEval) + np.sum(expCoef**2 / model.expEval)

    return w[0] * Elan + w[1] * Ereg

def initialShapeResiuals(param, target, model, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]

    # Insert z translation
    param = np.r_[param[:-1], 0, param[-1]]

    # Landmark fitting cost
    source = generateFace(param, model, ind = model.sourceLMInd)[:2, :]

    w0 = (w[0] / model.sourceLMInd.size)**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * (source - target.T).flatten('F'), w1 * idCoef ** 2 / model.idEval, w1 * expCoef ** 2 / model.expEval]

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
    # shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1)

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
    # shape = model.idMean[:, model.sourceLMInd] + np.tensordot(model.idEvec[:, model.sourceLMInd, :], idCoef, axes = 1)

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

    eq2 = np.zeros((idCoef.size, param.size))
    eq2[:, :idCoef.size] = np.diag(idCoef / model.idEval)

    eq3 = np.zeros((expCoef.size, param.size))
    eq3[:, idCoef.size : idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / model.sourceLMInd.size)**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * Jlan, w1 * eq2, w1 * eq3]


def cameraShapeCost(param, model, lm2d, lm3dInd, cam):
    """
    Minimize L2-norm of landmark fitting residuals and regularization terms for shape parameters
    """
    if cam == 'orthographic':
        param = param[:8]
        param = np.vstack((param.reshape((2, 4)), np.array([0, 0, 0, 1])))
        idCoef = param[8: 8 + model.numId]
        expCoef = param[8 + model.numId:]
    
    elif cam == 'perspective':
        param = param[:12]
        param = param.reshape((3, 4))
        idCoef = param[12: 12 + model.numId]
        expCoef = param[12 + model.numId:]
    
    # Convert to homogenous coordinates
    numLandmarks = lm3dInd.size
    
    lm3d = generateFace(np.r_[idCoef, expCoef, np.zeros(6), 1], model, ind = lm3dInd).T
    
    xlan = np.c_[lm2d, np.ones(numLandmarks)]
    Xlan = np.dot(np.c_[lm3d, np.ones(numLandmarks)], param.T)
    
    # Energy of landmark residuals
    rlan = (Xlan - xlan).flatten('F')
    Elan = np.dot(rlan, rlan)
    
    # Energy of shape regularization terms
    Ereg = np.sum(idCoef ** 2 / model.idEval) + np.sum(expCoef ** 2 / model.expEval)
    
    return Elan + Ereg

def textureCost(texCoef, img, vertexCoord, model, renderObj, w = (1, 1)):
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    # Color matching cost
    r = (rendering - img).flatten()
    Ecol = np.dot(r, r) / pixelCoord.shape[0]
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / model.texEval)

    return w[0] * Ecol + w[1] * Ereg

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

    w0 = (w[0] / numPixels)**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w[1] * texCoef ** 2 / model.texEval]

def textureGrad(texCoef, img, vertexCoord, model, renderObj, w = (1, 1)):
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

    return 2 * (w[0] * r.dot(J_texCoef) / numPixels + w[1] * texCoef / model.texEval)


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
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    w0 = (w[0] / numPixels)**(1/2)

    return np.r_[w0 * J_texCoef, w[1] * np.diag(texCoef / model.texEval)]

def textureLightingCost(texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), option = 'tl', constCoef = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    if option is 'tl':
        texCoef = texParam[:model.numTex]
        shCoef = texParam[model.numTex:].reshape(9, 3)
    elif option is 't':
        texCoef = texParam
        shCoef = constCoef.reshape(9, 3)
    elif option is 'l':
        texCoef = constCoef
        shCoef = texParam.reshape(9, 3)
    
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    # Color matching cost
    r = (rendering - img).flatten()
    Ecol = np.dot(r, r) / pixelCoord.shape[0]

    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / model.texEval)
    
    if option is 'l':
        return w[0] * Ecol
    else:
        return w[0] * Ecol + w[1] * Ereg

def textureLightingGrad(texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), option = 'tl', constCoef = None):
    if option is 'tl':
        texCoef = texParam[:model.numTex]
        shCoef = texParam[model.numTex:].reshape(9, 3)
    elif option is 't':
        texCoef = texParam
        shCoef = constCoef.reshape(9, 3)
    elif option is 'l':
        texCoef = constCoef
        shCoef = texParam.reshape(9, 3)
        
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
    numPixels = pixelFaces.size
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    pixelVertices = model.face[pixelFaces, :]
    
    r = rendering - img
    
    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    if option is 'tl':
        return 2 * w[0] * np.r_[r.flatten('F').dot(J_texCoef), r[:, 0].dot(J_shCoef[0]), r[:, 1].dot(J_shCoef[1]), r[:, 2].dot(J_shCoef[2])] / numPixels + np.r_[2 * w[1] * texCoef / model.texEval, np.zeros(27)]

    # Texture only
    elif option is 't':
        return 2 * (w[0] * r.flatten('F').dot(J_texCoef) / numPixels + w[1] * texCoef / model.texEval)
    
    # Light only
    elif option is 'l':
        return 2 * w[0] * np.r_[r[:, 0].dot(J_shCoef[0]), r[:, 1].dot(J_shCoef[1]), r[:, 2].dot(J_shCoef[2])] / numPixels
    
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

    w0 = (w[0] / numPixels)**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w[1] * texCoef ** 2 / model.texEval]

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
    
    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]
    
    w0 = (w[0] / numPixels)**(1/2)

    texCoefSide = np.r_[w0 * J_texCoef, w[1] * np.diag(texCoef / model.texEval)]
    shCoefSide = np.r_[w0 * block_diag(*J_shCoef), np.zeros((texCoef.size, shCoef.size))]

    return np.c_[texCoefSide, shCoefSide]

    
def denseResiduals(texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    # # Shape eigenvector coefficients
    # idCoef = param[: model.numId]
    # expCoef = param[model.numId: model.numId + model.numExp]

    # # Insert z translation
    # param = np.r_[param[:-1], 0, param[-1]]
    
    # Landmark fitting cost
    vertexCoords = generateFace(param, model, ind = model.sourceLMInd)[:2, :]

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
    
    return np.r_[w[0] / numPixels * (rendering - img).flatten('F'), w[1] * texCoef ** 2 / model.texEval]

def denseJacobian(texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), randomFaces = None):
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
    
    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]
    
    texCoefSide = np.r_[w[0] / numPixels * J_texCoef, w[1] * np.diag(texCoef / model.texEval)]
    shCoefSide = np.r_[w[0] / numPixels * block_diag(*J_shCoef), np.zeros((texCoef.size, shCoef.size))]
    
    return np.c_[texCoefSide, shCoefSide]