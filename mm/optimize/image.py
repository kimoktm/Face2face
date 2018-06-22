#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from scipy.interpolate import interp2d
from ..utils.mesh import calcNormals, generateFace, generateTexture, barycentricReconstruction
from ..utils.transform import rotMat2angle, sh9
from .derivative import dR_dpsi, dR_dtheta, dR_dphi, dR_normal, dR_sh

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

    xxx = np.zeros((pixelVertices.size, 27))
    for c in range(3):
        xxx[c*numPixels: (c+1)*numPixels, c * 9 : (c+1) * 9] = barycentricReconstruction(sh * vertexColor[c, :], pixelFaces, pixelBarycentricCoords, model.face)

    # print(np.array_equal(block_diag(*J_shCoef).shape, xxx))
    # print("##########")

    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(model.texEvec[c].T, pixelFaces, pixelBarycentricCoords, model.face)
        pixelSHLighting = barycentricReconstruction(np.dot(shCoef[:, c], sh), pixelFaces, pixelBarycentricCoords, model.face)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    w0 = (w[0] / numPixels)**(1/2)
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



def lightingResiduals(texParam, texCoef, img, vertexCoord, sh, model, renderObj, randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    shCoef = texParam.reshape(9, 3)

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

    r = (rendering - img).flatten()
    Ecol = np.dot(r, r) / pixelCoord.shape[0]

    # return Ecol
    return (rendering - img).flatten('F')

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

    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    J_shCoef = block_diag(*J_shCoef)

    xxx = np.zeros((pixelVertices.size, 27))
    for c in range(3):
        xxx[c*numPixels: (c+1)*numPixels, c * 9 : (c+1) * 9] = barycentricReconstruction(sh * vertexColor[c, :], pixelFaces, pixelBarycentricCoords, model.face)

    # np.savetxt('/home/karim/Desktop/arr2.csv', J_shCoef, delimiter=',')
    # print(xxx.shape)
    return xxx



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

    wcol = (w[0] / numPixels)**(1/2)
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

    wcol = (w[0] / numPixels)**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg_color = w[2]**(1/2)
    wreg_shape = w[3]**(1/2)

    # landmakrs error
    source = generateFace(shape_param, model, ind = model.sourceLMInd)[:2, :]

    # return np.r_[wcol * (rendering - img).flatten('F')]

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
    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, model.face)
    pixelSHBasis = barycentricReconstruction(sh, pixelFaces, pixelBarycentricCoords, model.face)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    J_shCoef = block_diag(*J_shCoef)

    # # Shape derivative
    # drV_dt = np.tile(np.eye(3), [model.numVertices, 1])
    # Klan_tmp = np.c_[drV_dalpha[:3, ...].reshape((vertexCoord.size, idCoef.size), order = 'F'), drV_ddelta[:3, ...].reshape((vertexCoord.size, expCoef.size), order = 'F'),\
    #  drV_dpsi[:3, :].flatten('F'), drV_dtheta[:3, :].flatten('F'), drV_dphi[:3, :].flatten('F'), drV_dt[:,:2], drV_ds[:3, :].flatten('F')]
    # Klan = np.reshape(Klan_tmp, (3, model.numVertices, param.size - model.numTex - 27))
    # xxx = dR_normal(vertexCoord, model, Klan)
    # zzz = dR_sh(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numTex - 27))
    for c in range(3):
        # lll = np.empty((zzz.shape[1:]))
        # for v in range(0, zzz.shape[2]):
        #     lll[:, v] = np.dot(shCoef[:, c], zzz[:,:,v])

        # shLighting = barycentricReconstruction(lll.T, pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] =  - imgDer

    # print(J_texCoef.shape)
    # print(J_shCoef.shape)
    # print(J_denshapeCoef.shape)

    # final dense jacobian (img_der * shape_der)
    J_denseCoef = np.empty((numPixels * 3, param.size))
    J_denseCoef = np.c_[J_texCoef, J_shCoef, J_denshapeCoef]
    # print(J_denseCoef.shape)
    # print("######################")

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

    Jlan_landmarks = np.c_[np.zeros((target.size, model.numTex + 27)), Jlan_landmarks]


    # weighting
    wcol = (w[0] / numPixels)**(1/2)
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

    # return np.r_[wcol * J_denseCoef]

    return np.r_[wcol * J_denseCoef, wlan * Jlan_landmarks, wreg_color * eq2, wreg_shape * eq3, wreg_shape * eq4]




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

    wcol = (w[0] / numPixels)**(1/2)
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
    wcol = (w[0] / numPixels)**(1/2)
    wlan = (w[1] / model.sourceLMInd.size)**(1/2)
    wreg = w[2]**(1/2)

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId))
    eq2[:, :expCoef.size] = np.diag(expCoef / model.expEval)

    return np.r_[wcol * -J_denseCoef, wlan * Jlan_landmarks, wreg * eq2]