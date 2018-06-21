#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from scipy.interpolate import interp2d
from ..utils.mesh import generateFace, generateTexture, barycentricReconstruction
from ..utils.transform import rotMat2angle
from .derivative import dR_dpsi, dR_dtheta, dR_dphi

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

    # Reg cost not correct
    return np.r_[w0 * (source - target.T).flatten('F'), w1 * idCoef ** 2 / model.idEval, w1 * expCoef ** 2 / model.expEval]

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

    # print(pixelVertices.shape)
    # print(pixelVertices.size)
    # print(numPixels)
    # print(model.texEvec.T.shape)
    # print('----------------')

    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        J_texCoef[c * numPixels: (c + 1) * numPixels, :] = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)

    w0 = (w[0] / numPixels)**(1/2)
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

    w0 = (w[0] / numPixels)**(1/2)
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

    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)

    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    w0 = (w[0] / numPixels)**(1/2)
    w1 = w[1]**(1/2)

    texCoefSide = np.r_[w0 * J_texCoef, w1 * np.diag(texCoef / model.texEval)]
    shCoefSide = np.r_[w0 * block_diag(*J_shCoef), np.zeros((texCoef.size, shCoef.size))]

    # print(J_texCoef.shape)
    # print(block_diag(*J_shCoef).shape)
    # print(texCoefSide.shape)
    # print(shCoefSide.shape)
    # print( np.c_[texCoefSide, shCoefSide].shape)

    return np.c_[texCoefSide, shCoefSide]



def denseResiduals(param, img, texCoef, model, renderObj, w = (1, 1), randomFaces = None):
    # Shape eigenvector coefficients
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

    w0 = (w[0] / numPixels)**(1/2)
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
    Jlan = np.reshape(Jlan_tmp, (2, model.numVertices, param.size))
    
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

    # Reg cost not correct
    eq2 = np.zeros((idCoef.size, param.size))
    eq2[:, :idCoef.size] = np.diag(idCoef / model.idEval)

    eq3 = np.zeros((expCoef.size, param.size))
    eq3[:, idCoef.size : idCoef.size + expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / numPixels)**(1/2)
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

    w0 = (w[0] / numPixels)**(1/2)
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

    w0 = (w[0] / numPixels)**(1/2)
    w1 = w[1]**(1/2)
    w2 = w[2]**(1/2)

    return np.r_[w0 * J_denseCoef, w1 * eq2, w2 * eq3, w2 * eq4]



def denseAllResiduals(param, img, target, model, renderObj, w = (1, 1, 1, 1), randomFaces = None):
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

    wcol = (w[0] / numPixels)**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # landmakrs error
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    return np.r_[wcol * (rendering - img).flatten('F'), wlan * (source - target.T).flatten('F'), wreg_color * texCoef ** 2 / model.texEval, wreg_shape * idCoef ** 2 / model.idEval, wreg_shape * expCoef ** 2 / model.expEval]


def denseAllJacobian(param, img, target, model, renderObj, w = (1, 1, 1, 1), randomFaces = None):
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

    w0 = (w[0] / numPixels)**(1/2)
    w1 = w[1]**(1/2)
    w2 = w[2]**(1/2)


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
    wcol = (w[0] / numPixels)**(1/2)
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



def denseExpResiduals(param, idCoef, img, texCoef, model, renderObj, w = (1, 1), randomFaces = None):
    param = np.r_[idCoef, param]
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

    w0 = (w[0] / numPixels)**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * (rendering - img).flatten('F'), w1 * expCoef ** 2 / model.expEval]

def denseExpJacobian(param, idCoef, img, texCoef, model, renderObj, w = (1, 1), randomFaces = None):
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

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId))
    eq2[:, :expCoef.size] = np.diag(expCoef / model.expEval)

    w0 = (w[0] / numPixels)**(1/2)
    w1 = w[1]**(1/2)

    return np.r_[w0 * -J_denseCoef, w1 * eq2]