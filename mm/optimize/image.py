#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from ..utils.mesh import calcNormals, calcFaceNormals, generateFace, generateTexture, barycentricReconstruction
from ..utils.transform import rotMat2angle, sh9
from .derivative import dR_dpsi, dR_dtheta, dR_dphi, dR_normal, dR_normal_faces, dR_sh


## CORRECT
def initialShapeResiduals(param, target, model, w = (1, 1)):
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

def expResiduals(param, idCoef, target, model, w = (1, 1)):
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
    # Use per face normal which easier to differentiate
    faceNorms = calcFaceNormals(vertexCoord, model)
    xxx = dR_normal_faces(vertexCoord, model, Jlan)
    zzz = dR_sh(faceNorms[:, 0], faceNorms[:, 1], faceNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])
    lll = np.empty((3, zzz.shape[1], zzz.shape[2]))
    for c in range(3):
        for v in range(0, zzz.shape[2]):
            lll[c, :, v] = np.dot(shCoef[:, c], zzz[:, :, v])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numTex - 27))
    for c in range(3):
        # go to pixel space
        shLighting = lll[c, pixelFaces] * barycentricReconstruction(vertexColor[c, :], pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] = shLighting - imgDer


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
    for l in range(2):
        J_shapeCoef[l, :, :] = barycentricReconstruction(Jlan[l].T, pixelFaces, pixelBarycentricCoords, model.face)

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
    # Use per face normal which easier to differentiate
    faceNorms = calcFaceNormals(vertexCoord, model)
    xxx = dR_normal_faces(vertexCoord, model, Jlan)
    zzz = dR_sh(faceNorms[:, 0], faceNorms[:, 1], faceNorms[:, 2], xxx[:, 0], xxx[:, 1], xxx[:, 2])
    lll = np.empty((3, zzz.shape[1], zzz.shape[2]))
    for c in range(3):
        for v in range(0, zzz.shape[2]):
            lll[c, :, v] = np.dot(shCoef[:, c], zzz[:, :, v])

    J_denshapeCoef = np.empty((numPixels * 3, param.size - model.numId))
    for c in range(3):
        # go to pixel space
        shLighting = lll[c, pixelFaces] * barycentricReconstruction(vertexColor[c, :], pixelFaces, pixelBarycentricCoords, model.face)
        imgDer = np.multiply(J_shapeCoef[0, :, :], img_grad_x[:, c][:, np.newaxis]) + np.multiply(J_shapeCoef[1, :, :], img_grad_y[:, c][:, np.newaxis])
        J_denshapeCoef[c * numPixels: (c + 1) * numPixels, :] = shLighting - imgDer

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

    # REMOVE - TESTING
    # Fix non-constant sh coff (dependent on normals)
    # J_shCoef[:, 3:] = 0

    J_denseCoef = np.c_[J_shCoef, J_denshapeCoef]

    # Reg cost not correct
    eq2 = np.zeros((expCoef.size, param.size - model.numId + 27))
    eq2[:, 27 : 27 + expCoef.size] = np.diag(expCoef / model.expEval)

    return np.r_[wcol * J_denseCoef, wlan * Jlan_landmarks, wreg_shape * eq2]



# MULTI-FRAME
def multiDenseJointResiduals(params, imgs, targets, model, renderObj, w = (1, 1, 1, 1), randomFacesNum = None):
    if len(imgs.shape) is 3:
        imgs = imgs[np.newaxis, :]

    if len(targets.shape) is 2:
        targets = targets[np.newaxis, :]

    num_images = imgs.shape[0]
    unique_params = int((params.size - (model.numTex + model.numId)) / num_images)

    # texCoef, idCoef are fixed across all images
    # params (texCoef, idCoef, (27 sh, 78 expCoef, 6 poseCoef) * num images)
    texCoef = params[: model.numTex]
    idCoef = params[model.numTex : model.numTex + model.numId]

    residuals = np.array([])
    for i in range(num_images):
        shRange  = np.index_exp[model.numTex + model.numId + unique_params * i : model.numTex + model.numId + unique_params * i + 27]
        expRange = np.index_exp[model.numTex + model.numId + 27 + unique_params * i : model.numTex + model.numId + unique_params * i + unique_params]

        shCoef = params[shRange]
        expCoef = params[expRange]
        param = np.r_[texCoef, shCoef, idCoef, expCoef]
        img_residuals = denseJointResiduals(param, imgs[i], targets[i], model, renderObj, w, randomFacesNum)
        residuals = np.r_[residuals, img_residuals] if residuals.size else img_residuals

    return residuals

def multiDenseJointJacobian(params, imgs, targets, model, renderObj, w = (1, 1, 1, 1), randomFacesNum = None):
    if len(imgs.shape) is 3:
        imgs = imgs[np.newaxis, :]

    if len(targets.shape) is 2:
        targets = targets[np.newaxis, :]

    num_images = imgs.shape[0]
    unique_params = int((params.size - (model.numTex + model.numId)) / num_images)

    # texCoef, idCoef are fixed across all images
    # params (texCoef, idCoef, (27 sh, 78 expCoef, 6 poseCoef) * num images)
    texCoef = params[: model.numTex]
    idCoef = params[model.numTex : model.numTex + model.numId]

    jacobians = np.array([])
    for i in range(num_images):
        shRange  = np.index_exp[model.numTex + model.numId + unique_params * i : model.numTex + model.numId + unique_params * i + 27]
        expRange = np.index_exp[model.numTex + model.numId + 27 + unique_params * i : model.numTex + model.numId + unique_params * i + unique_params]

        shCoef = params[shRange]
        expCoef = params[expRange]
        param = np.r_[texCoef, shCoef, idCoef, expCoef]
        img_jacobian = denseJointJacobian(param, imgs[i], targets[i], model, renderObj, w, randomFacesNum)

        # Need to re-arrange jacobian correctly
        extended_jacobian = np.zeros((img_jacobian.shape[0], params.size))
        extended_jacobian[:, : model.numTex] = img_jacobian[:, : model.numTex] # texCoef
        extended_jacobian[:, model.numTex : model.numTex + model.numId] = img_jacobian[:, model.numTex + 27 : model.numTex + 27 + model.numId] # idCoef
        extended_jacobian[:, model.numTex + model.numId + unique_params * i : model.numTex + model.numId + unique_params * i + 27] = img_jacobian[:, model.numTex : model.numTex + 27] # img_shCoef
        extended_jacobian[:, model.numTex + model.numId + 27 + unique_params * i : model.numTex + model.numId + unique_params * i + unique_params] = img_jacobian[:, model.numTex + 27 + model.numId :] # img_expCoef
        jacobians = np.r_[jacobians, extended_jacobian] if jacobians.size else extended_jacobian

    return jacobians
