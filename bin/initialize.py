#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estimateCamMat, splitCamMat
import mm.optimize.image as opt
from mm.utils.mesh import calcNormals, generateFace, generateTexture, barycentricReconstruction, writePly
from mm.utils.transform import sh9

import os, json
import numpy as np
from scipy.optimize import minimize, check_grad, least_squares, nnls, lsq_linear
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import savefig

from scipy.special import sph_harm
from mayavi import mlab
import time
import dlib
import cv2


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    #detekcja twarzy
    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))

        #detekcja punktow charakterystycznych twarzy
        dlibShape = predictor(img, faceRectangle)
        
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        #transpozycja, zeby ksztalt byl 2 x n a nie n x 2, pozniej ulatwia to obliczenia
        shape2D = shape2D.T

        shapes2D.append(shape2D)

    return shapes2D



if __name__ == "__main__":
    
    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('./data')
    
    # Input the number of frames in the video
    numFrames = 995 #2260 #3744
    
    # Load 3DMM
    m = MeshModel('../models/bfm2017.npz')
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'

    # Landmark detector
    predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Set weights for the 3DMM RGB color shape, landmark shape, and regularization terms
    wCol = 1
    wLan = 1
    # wReg = 2.5e-05
    wReg = 0.01

    for frame in np.arange(0, 1):
        print(frame)
        fName = 'subject_1_{:0>12}'.format(frame)
        
        """
        Set filenames, read landmarks, load source video frames
        """
        # Frames from the source video
        fNameImgOrig = 'orig/' + fName + '_rendered.png'

        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(fNameImgOrig))
        img_org = cv2.merge([r,g,b])
        img = img_as_float(img_org)

        shape2D = getFaceKeypoints(img_org, detector, predictor)
        shape2D = np.asarray(shape2D)[0].T 
        lm = shape2D[m.targetLMInd, :2]

        """
        Initial registration of similarity transform and shape coefficients
        """        
        # Initialize 3DMM parameters for the first frame
        if frame == 0:
            idCoef = np.zeros(m.numId)
            expCoef = np.zeros(m.numExp)
            texCoef = np.zeros(m.numTex)
            param = np.r_[np.zeros(m.numId + m.numExp + 6), 1]
        
        # Get the vertex values of the 3DMM landmarks
        lm3D = generateFace(param, m, ind = m.sourceLMInd).T


        vertexCoords = generateFace(param, m)
        writePly("../mesh.ply", vertexCoords, m.face, m.texMean, m.sourceLMInd)

        # Estimate the camera projection matrix from the landmark correspondences
        camMat = estimateCamMat(lm, lm3D, cam)
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(camMat, cam)
        
        # Concatenate parameters for input into optimization routine. Note that the translation vector here is only (2,) for x and y (no z)
        param = np.r_[idCoef, expCoef, angles, t, s]


        # Initial optimization of shape parameters with similarity transform parameters
        # initFit = least_squares(opt.initialShapeResiuals, param, jac = opt.initialShapeJacobians, args = (lm, m, (wLan, wReg)), loss = 'linear', verbose = 1)
        # param = initFit['x']

        idCoef = param[:m.numId]
        expCoef = param[m.numId: m.numId+m.numExp]
        
        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # Rendering of initial 3DMM shape with mean texture model
        texture = m.texMean
        meshData = np.r_[vertexCoords.T, texture.T]
        renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)
        renderObj.render()

        # Grab the OpenGL rendering from the video card
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        plt.figure("fitting")
        plt.imshow(img)
        plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')

        # Plot the OpenGL rendering
        plt.figure("rendering")
        plt.imshow(rendering)


        # """
        # Get initial texture parameter guess
        # """
        
        # # Set the number of faces for stochastic optimization
        numRandomFaces = 10000
        # start = time.time()

        # wReg = 0.00005

        # # Do some cycles of nonlinear least squares iterations, using a new set of random faces each time for the optimization objective
        # for i in range(1):
        #     randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     initTex = least_squares(opt.textureResiduals, texCoef, jac = opt.textureJacobian, args = (img, vertexCoords, m, renderObj, (wCol, wReg), randomFaces), loss = 'soft_l1', verbose = 1)
        #     texCoef = initTex['x']

        # elapsed = time.time() - start
        # print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        
        # # Update the rendering and plot
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # # writePly("../mesh_landmarks.ply", vertexCoords, m.face, texture)

        # plt.figure("Texture")
        # plt.imshow(rendering)
        # # plt.show()


        # # """
        # # Optimization over dense shape
        # # """
        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # initFit = least_squares(opt.denseResiduals, param, jac = opt.denseJacobian, args = (img, texCoef, m, renderObj, (wLan, 0.000000)), loss = 'linear', verbose = 1)
        # param = initFit['x']

        # # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # plt.figure("Dense Shape")
        # plt.imshow(rendering)


        # writePly("../mesh.ply", vertexCoords, m.face, texture)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()


        # # """
        # # Optimization over dense shape & albedo
        # # """
        # param = np.r_[texCoef, param]

        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # initFit = least_squares(opt.denseTexResiduals, param, jac = opt.denseTexJacobian, args = (img, m, renderObj, (wLan, 0.00005, 0.000001)), loss = 'linear', verbose = 1)
        # param_all = initFit['x']
        # texCoef = param_all[:m.numTex]
        # param = param_all[m.numTex:]

        # # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # plt.figure("Dense Shape")
        # plt.imshow(rendering)

        # writePly("../mesh.ply", vertexCoords, m.face, texture)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()


        # """
        # Optimization over all dense shape
        # """
        param = np.r_[texCoef, param]

        # Jointly optimize the texture and spherical harmonic lighting coefficients
        initFit = least_squares(opt.denseAllResiduals, param, jac = opt.denseAllJacobian, args = (img, lm, m, renderObj, (1, 6, 0.0002, 0.08)), loss = 'linear', verbose = 2)
        param_all = initFit['x']
        texCoef = param_all[:m.numTex]
        param = param_all[m.numTex:]

        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # Generate the texture at the 3DMM vertices from the learned texture coefficients
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)

        # Render the 3DMM
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        plt.figure("Dense Shape 1")
        plt.imshow(rendering)

        writePly("../mesh_step1.ply", vertexCoords, m.face, texture)

        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()


        """
        Get initial spherical harmonic lighting parameter guess
        """
        # Calculate normals at each vertex in the 3DMM
        vertexNorms = calcNormals(vertexCoords, m)

        # Evaluate spherical harmonics at face shape normals. The result is a (numVertices, 9) array where each column is a spherical harmonic basis for the 3DMM.
        B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        
        # Get the pixel RGB values of the original image where the 3DMM face is rendered
        imgMasked = img[pixelCoord[:, 0], pixelCoord[:, 1]]
        
        # Initialize an array to store the barycentric reconstruction of the nine spherical harmonics. The first dimension indicates the color (RGB).
        I = np.empty((3, pixelFaces.size, 9))
        
        # Initialize an array to store the spherical harmonic lighting coefficients. There are nine coefficients per color channel.
        shCoef = np.empty((9, 3))
        
        # Loop through each color channel
        for c in range(3):
            # 
            I[c, ...] = barycentricReconstruction(B * texture[c, :], pixelFaces, pixelBarycentricCoords, m.face)

            # Make an initial guess of the spherical harmonic lighting coefficients with least squares. We are solving Ax = b, where A is the (numFaces, 9) array of the barycentric reconstruction of the spherical harmonic bases, x is the (9,) vector of coefficients, and b is the (numFaces,) vector of the pixels from the original image where the 3DMM is defined.
#            shCoef[:, c] = nnls(I[c, ...], imgMasked[:, c])[0]
            shCoef[:, c] = lsq_linear(I[c, ...], imgMasked[:, c]).x

        # Concatenate the texture coefficients with the spherical harmonic coefficients and use a helper function to generate the RGB values at each vertex on the 3DMM
        texParam = np.r_[texCoef, shCoef.flatten()]
        textureWithLighting = generateTexture(vertexCoords, texParam, m)
        
        # Render the 3DMM with the initial guesses for texture and lighting
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, textureWithLighting.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        plt.figure("SH")
        plt.imshow(rendering)



        # """
        # Optimization over all dense shape with SH fixed
        # """
        param = np.r_[texCoef, param]

        # Jointly optimize the texture and spherical harmonic lighting coefficients
        initFit = least_squares(opt.denseShResiduals, param, jac = opt.denseShJacobian, args = (img, shCoef.flatten(), lm, m, renderObj, (1, 6, 0.0002, 0.08)), loss = 'linear', verbose = 2)
        param_all = initFit['x']
        texCoef = param_all[:m.numTex]
        param = param_all[m.numTex:]

        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # Generate the texture at the 3DMM vertices from the learned texture coefficients
        texParam = np.r_[texCoef, shCoef.flatten()]
        texture = generateTexture(vertexCoords, texParam, m)

        # Render the 3DMM
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        plt.figure("Dense Shape 2")
        plt.imshow(rendering)

        writePly("../mesh_step2.ply", vertexCoords, m.face, texture)
        np.save("parameters", np.r_[texCoef, shCoef.flatten(), param])

        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        plt.figure("Desne fitting 2")
        plt.imshow(img)
        plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        plt.show()


        # """
        # Optimization over shape, texture, and lighting
        # """
        # # shCoef = np.zeros((9, 3))
        # # shCoef[0, 0] = 0.5
        # # shCoef[0, 1] = 0.5
        # # shCoef[0, 2] = 0.5
        # allParam = np.r_[texCoef, shCoef.flatten(), param]

        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # randomFacesNum = 10000
        # initShapeTexLight = least_squares(opt.denseJointResiduals, allParam, jac = opt.denseJointJacobian, args = (img, lm, m, renderObj, (1, 6, 0.0002, 0.08), randomFacesNum), loss = 'linear', verbose = 2)
        # allParam = initShapeTexLight['x']

        # print(texCoef.size + shCoef.size)
        # texParam3 = allParam[:texCoef.size + shCoef.size]
        # shapeParam3 = allParam[texCoef.size + shCoef.size:]

        # # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[shapeParam3[:-1], 0, shapeParam3[-1]], m)

        # # Generate 3DMM texture form vertex & sh parameters
        # texture = generateTexture(vertexCoords, texParam3, m)
        # # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # print(texParam3[texCoef.size:].reshape(9, 3))

        # plt.figure("Shape & Texture & SH")
        # plt.imshow(rendering)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        
        # writePly("../mesh_sh_all.ply", vertexCoords, m.face, texture)
        # plt.show()




        # """
        # Optimization simultaneously over lighting parameters
        # """

        # shCoef = np.ones((9, 3)) * 0.0
        # shCoef[0, 0] = 0.5
        # shCoef[0, 1] = 0.5
        # shCoef[0, 2] = 0.5
        # print(shCoef)
        # # Calculate normals at each vertex in the 3DMM
        # vertexNorms = calcNormals(vertexCoords, m)

        # # Evaluate spherical harmonics at face shape normals. The result is a (numVertices, 9) array where each column is a spherical harmonic basis for the 3DMM.
        # B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

        # texParam2 = shCoef.flatten()

        # wReg = 0
        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # for i in range(1):
        #     randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     # initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, vertexCoords, B, m, renderObj, (wCol, wReg)), loss = 'soft_l1', max_nfev = 100)
        #     # initFit = minimize(opt.lightingResiduals, texParam2, args = (texCoef, img, vertexCoords, B, m, renderObj), options={'disp': True}, method = 'BFGS')
        #     # texParam2 = initFit.x
        #     initTexLight = least_squares(opt.lightingResiduals, texParam2, jac = opt.lightingJacobian, args = (texCoef, img, vertexCoords, B, m, renderObj), loss = 'linear', method = 'trf', tr_solver = 'lsmr', verbose = 2)
        #     texParam2 = initTexLight['x']

        # shCoef = texParam2.reshape(9, 3)

        # texture = generateTexture(vertexCoords, np.r_[texCoef, shCoef.flatten()], m)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # print(shCoef)
        # plt.figure("Texture & SH")
        # plt.imshow(rendering)

        # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # plt.figure("Texture Only")
        # plt.imshow(rendering)
        # plt.show()


        # """
        # Optimization simultaneously over the texture and lighting parameters
        # """

        # shCoef = np.ones((9, 3)) * 0.0
        # shCoef[0, 0] = 0.5
        # shCoef[0, 1] = 0.5
        # shCoef[0, 2] = 0.5
        # print(shCoef)
        # texParam = np.r_[texCoef, shCoef.flatten()]
        # # Calculate normals at each vertex in the 3DMM
        # vertexNorms = calcNormals(vertexCoords, m)

        # # Evaluate spherical harmonics at face shape normals. The result is a (numVertices, 9) array where each column is a spherical harmonic basis for the 3DMM.
        # B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

        # texParam2 = texParam.copy()

        # wReg = 0
        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # for i in range(1):
        #     randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, vertexCoords, B, m, renderObj, (wCol, wReg)), loss = 'linear')
        #     # initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, vertexCoords, B, m, renderObj, (wCol, wReg)), loss = 'soft_l1', max_nfev = 100)
        #     texParam2 = initTexLight['x']

        # texCoef = texParam2[:m.numTex]
        # shCoef = texParam2[m.numTex:].reshape(9, 3)

        # texture = generateTexture(vertexCoords, texParam2, m)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # print(shCoef)
        # plt.figure("Texture & SH")
        # plt.imshow(rendering)

        # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # plt.figure("Texture Only")
        # plt.imshow(rendering)

        # plt.show()



        # """
        # Optimization over shape, texture, and lighting
        # """
        # allParam = np.r_[param.copy(), texParam.copy()]
        # # allParam[:m.numId + m.numExp + 6] = param.copy()
        # # allParam[m.numId + m.numExp + 6:] = texParam.copy()

        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # cost = np.zeros(10)
        # for i in range(1):
        #     randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     initShapeTexLight = least_squares(opt.shapeTextureLightingResiduals, allParam, args = (img, m, renderObj, (wCol, wReg)), loss = 'soft_l1')
        #     allParam = initShapeTexLight['x']
        #     cost[i] = initShapeTexLight.cost

        # shapeParam3 = allParam[:param.size]
        # texParam3 = allParam[param.size:]

        # # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[shapeParam3[:-1], 0, shapeParam3[-1]], m)

        # # Generate 3DMM texture form vertex & sh parameters
        # texture = generateTexture(vertexCoords, texParam3, m)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        # plt.figure("Shape & Texture & SH")
        # plt.imshow(rendering)
        # plt.show()