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
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import savefig

import scipy.misc
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
    startFrame = 0
    numFrames = 300 #2260 #3744
    
    # Load 3DMM
    m = MeshModel('../models/bfm2017.npz')
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'

    # Landmark detector
    predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Load parameters
    all_param = np.load("./obama_parameters.npy")
    texCoef = all_param[:m.numTex]
    shCoef = all_param[m.numTex: m.numTex + 27]
    param = all_param[m.numTex + 27:]
    idCoef = param[:m.numId]
    expCoef = param[m.numId : m.numId + m.numExp]

    for frame in np.arange(startFrame, numFrames):
        print(frame)

        fName = '{:0>6}'.format(frame)
        fNameImgOrig = 'obama/orig/' + fName + '.png'

        # fName = 'subject_1_{:0>12}'.format(frame)
        # fNameImgOrig = 'jack/orig/' + fName + '_rendered.png'

        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(fNameImgOrig))
        img_org = cv2.merge([r,g,b])
        # img_org = cv2.GaussianBlur(img_org, (9,9), 0)
        img = img_as_float(img_org)

        # plt.figure("Blurre")
        # plt.imshow(img)
        # plt.show()

        shape2D = getFaceKeypoints(img_org, detector, predictor)
        shape2D = np.asarray(shape2D)[0].T 
        lm = shape2D[m.targetLMInd, :2]

        if frame == startFrame:
            vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
            # Rendering of initial 3DMM shape with mean texture model
            texParam = np.r_[texCoef, shCoef.flatten()]
            texture = generateTexture(vertexCoords, texParam, m)
            meshData = np.r_[vertexCoords.T, texture.T]
            renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)
            renderObj.render()

            # Grab the OpenGL rendering from the video card
            rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

            # plt.figure("Initial")
            # plt.imshow(rendering)
            scipy.misc.imsave("./" + str(frame) + "_orig.png", rendering)



        # """
        # Optimization over all experssion & SH
        # """


        # Landmarks fitting - Should be removed
        # initFit = least_squares(opt.expResiduals, param[m.numId:], jac = opt.expJacobians, args = (idCoef, lm, m, (1, 0.05)), x_scale = 'jac')
        # param[m.numId:] = initFit['x']
        # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texParam = np.r_[texCoef, shCoef.flatten()]
        # texture = generateTexture(vertexCoords, texParam, m)
        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # plt.figure("Sparse 1")
        # plt.imshow(rendering)
        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Sparse fitting 1")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()


        # Coarse to fine optimization
        # iterations = np.array([5, 3, 2])
        # for i in iterations:

        # scale_factor = 4
        # img_resized = resize(img, (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)))
        # param[-3:] = param[-3:] / scale_factor
        # renderObj = Render(img_resized.shape[1], img_resized.shape[0], meshData, m.face)
        # initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img_resized, lm / scale_factor, m, renderObj, (2, 10 * scale_factor, 0.000000005)), loss = 'linear', verbose = 0, max_nfev = 5, method = 'trf', tr_solver='lsmr')
        # shCoef = initFit['x'][:27]
        # expCoef = initFit['x'][27:]
        # param = np.r_[idCoef, expCoef]
        # # Generate 3DMM vertices from shape and similarity transform parameters
        # # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # # texParam = np.r_[texCoef, shCoef.flatten()]
        # # texture = generateTexture(vertexCoords, texParam, m)
        # # # Render the 3DMM
        # # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # # renderObj.resetFramebufferObject()
        # # renderObj.render()
        # # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # # plt.figure("Dense Shape 1")
        # # plt.imshow(rendering)
        # # Upscale paramters again
        # param[-3:] = param[-3:] * scale_factor
        # renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)


        # scale_factor = 2
        # img_resized = resize(img, (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)))
        # param[-3:] = param[-3:] / scale_factor
        # renderObj = Render(img_resized.shape[1], img_resized.shape[0], meshData, m.face)
        # initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img_resized, lm / scale_factor, m, renderObj, (2, 10 * scale_factor, 0.000000005)), loss = 'linear', verbose = 0, max_nfev = 3, method = 'trf', tr_solver='lsmr')
        # shCoef = initFit['x'][:27]
        # expCoef = initFit['x'][27:]
        # param = np.r_[idCoef, expCoef]
        # # Generate 3DMM vertices from shape and similarity transform parameters
        # # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # # texParam = np.r_[texCoef, shCoef.flatten()]
        # # texture = generateTexture(vertexCoords, texParam, m)
        # # # Render the 3DMM
        # # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # # renderObj.resetFramebufferObject()
        # # renderObj.render()
        # # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # # plt.figure("Dense Shape 2")
        # # plt.imshow(rendering)
        # # Upscale paramters again
        # param[-3:] = param[-3:] * scale_factor
        # renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)


        # numRandomFaces =  10000
        # randomFaces = np.random.randint(0, pixelFaces.size - 300, numRandomFaces)
        # initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img, lm, m, renderObj, (1, 0, 0.0)), loss = 'linear', verbose = 2, max_nfev = 30, tr_solver = 'lsmr', method = 'trf')
        # shCoef = initFit['x'][:27]
        # expCoef = initFit['x'][27:]
        # param = np.r_[idCoef, expCoef]



        # scale_factor = 4
        # img_resized = resize(img, (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)))
        # param[-3:] = param[-3:] / scale_factor
        # renderObj = Render(img_resized.shape[1], img_resized.shape[0], meshData, m.face)
        # initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], max_nfev = 5, jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img_resized, lm, m, renderObj, (1, 0, 2.5e-4)), verbose = 0, x_scale = 'jac')
        # shCoef = initFit['x'][:27]
        # expCoef = initFit['x'][27:]
        # param = np.r_[idCoef, expCoef]
        # # # Render the 3DMM
        # # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # # renderObj.resetFramebufferObject()
        # # renderObj.render()
        # # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # # plt.figure("Dense Shape 1")
        # # plt.imshow(rendering)
        # param[-3:] = param[-3:] * scale_factor

        # scale_factor = 2
        # img_resized = resize(img, (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)))
        # param[-3:] = param[-3:] / scale_factor
        # renderObj = Render(img_resized.shape[1], img_resized.shape[0], meshData, m.face)
        # initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], max_nfev = 3, jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img_resized, lm, m, renderObj, (1, 0, 2.5e-4)), verbose = 0, x_scale = 'jac')
        # shCoef = initFit['x'][:27]
        # expCoef = initFit['x'][27:]
        # param = np.r_[idCoef, expCoef]
        # # # Render the 3DMM
        # # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # # renderObj.resetFramebufferObject()
        # # renderObj.render()
        # # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # # plt.figure("Dense Shape 2")
        # # plt.imshow(rendering)
        # param[-3:] = param[-3:] * scale_factor
        # renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)


        # initFit = least_squares(opt.denseExpOnlyResiduals, param[m.numId:], jac = opt.denseExpOnlyJacobian, max_nfev = 10, args = (idCoef, texCoef, shCoef.reshape(9 ,3), img, m, renderObj, (1, 2.5e-4)), verbose = 0, x_scale = 'jac')
        # expCoef = initFit['x']
        # param = np.r_[idCoef, expCoef]

        # initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], max_nfev = 10, jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img, lm, m, renderObj, (1, 0.000025, 2.5e-4)), verbose = 0, x_scale = 'jac')
        # LSMR is numerically stable combared to the default option (Exact)
        #
        initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], tr_solver = 'lsmr', max_nfev = 10, jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img, lm, m, renderObj, (1, 2.5e-5, 1.25e-4)), verbose = 0, x_scale = 'jac')
        shCoef = initFit['x'][:27]
        expCoef = initFit['x'][27:]
        param = np.r_[idCoef, expCoef]

        # # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # Generate the texture at the 3DMM vertices from the learned texture coefficients
        texParam = np.r_[texCoef, shCoef.flatten()]
        texture = generateTexture(vertexCoords, texParam, m)

        # Render the 3DMM
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # plt.figure("Dense Shape 3")
        # plt.imshow(rendering)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting 3")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()

        scipy.misc.imsave("./" + str(frame) + ".png", rendering)
        #np.save("./" + str(frame) + "_parameters", param)
        # break
