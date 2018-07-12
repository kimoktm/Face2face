#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estimateCamMat, splitCamMat
import mm.optimize.image as opt
from mm.utils.mesh import generateFace, generateTexture, writePly

import os
import time
import dlib
import cv2
import scipy.misc
import numpy as np
from scipy.optimize import least_squares
from skimage import io, img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt



def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))
        dlibShape = predictor(img, faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D



if __name__ == "__main__":
    
    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('./data')

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
    wLan = 1.25e-4
    wRegC = 2.5e-5
    wRegS = 1.25e-5


    # load images
    idCoef = np.zeros(m.numId)
    expCoef = np.zeros(m.numExp)
    texCoef = np.zeros(m.numTex)
    shCoef = np.zeros((9, 3))
    shCoef[0, 0] = 0.5
    shCoef[0, 1] = 0.5
    shCoef[0, 2] = 0.5

    imgs = []
    landmarks = []
    img_params = []
    renderObj = None

    for frame in np.arange(0, 3):
        fName = '{:0>6}'.format(frame)
        fNameImgOrig = 'obama/init/' + fName + '.png'

        # fName = 'subject_1_{:0>12}'.format(frame)
        # fNameImgOrig = 'jack/orig/' + fName + '_rendered.png'

        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(fNameImgOrig))
        img_org = cv2.merge([r,g,b])
        img = img_as_float(img_org)

        shape2D = getFaceKeypoints(img_org, detector, predictor)
        shape2D = np.asarray(shape2D)[0].T 
        lm = shape2D[m.targetLMInd, :2]

        # Resize image for speed
        scale_factor = 2.0
        img = resize(img, (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)))
        lm = lm / scale_factor

        """
        Initial registration of similarity transform and shape coefficients
        """        
        param = np.r_[np.zeros(m.numId + m.numExp + 6), 1]
        
        # Get the vertex values of the 3DMM landmarks
        lm3D = generateFace(param, m, ind = m.sourceLMInd).T

        vertexCoords = generateFace(param, m)

        # Estimate the camera projection matrix from the landmark correspondences
        camMat = estimateCamMat(lm, lm3D, cam)
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(camMat, cam)

        # Concatenate parameters for input into optimization routine. Note that the translation vector here is only (2,) for x and y (no z)
        unique_img_param = np.r_[shCoef.flatten(), expCoef, angles, t, s]

        if frame == 0:
            # Initialize render Object
            texture = m.texMean
            meshData = np.r_[vertexCoords.T, texture.T]
            renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)

        # append parameters
        imgs.append(img)
        landmarks.append(lm)
        img_params.append(unique_img_param)

    imgs = np.asarray(imgs)
    landmarks = np.asarray(landmarks)
    img_params = np.asarray(img_params)

    if len(imgs.shape) is 3:
        imgs = imgs[np.newaxis, :]
        landmarks = landmarks[np.newaxis, :]
        img_params = img_params[np.newaxis, :]




    #
    # Jointly optimize all params over N images
    #
    start = time.time()

    allParams = np.r_[texCoef, idCoef, img_params.flatten()]
    initShapeTexLight = least_squares(opt.multiDenseJointResiduals, allParams, jac = opt.multiDenseJointJacobian, tr_solver = 'lsmr', max_nfev = 80, args = (imgs, landmarks, m, renderObj, (wCol, wLan, wRegC, wRegS)), verbose = 2, x_scale = 'jac')
    allParams = initShapeTexLight['x']
    texCoef = allParams[: texCoef.size]
    idCoef = allParams[texCoef.size : texCoef.size + idCoef.size]
    img_params = allParams[texCoef.size + idCoef.size :].reshape(img_params.shape)

    elapsed = time.time() - start
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    # Visualize results
    for i in range(img_params.shape[0]):
        shCoef = img_params[i, : 27]
        expCoef = img_params[i, 27 : ]
        texParam = np.r_[texCoef, shCoef]
        shapeParam = np.r_[idCoef, expCoef]

        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[shapeParam[:-1], 0, shapeParam[-1]], m)

        # Generate 3DMM texture form vertex & sh parameters
        texture = generateTexture(vertexCoords, texParam, m)

        # Render the 3DMM
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # print(texParam[texCoef.size:].reshape(9, 3))
        plt.figure(str(i) + " Shape & Texture & SH")
        plt.imshow(rendering)

        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        plt.figure(str(i) + " Desne fitting")
        plt.imshow(imgs[i])
        plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        plt.scatter(landmarks[i, :, 0], landmarks[i, :, 1], s = 2, c = 'g')

        writePly("../mesh_multi_all.ply", vertexCoords, m.face, texture)
        scipy.misc.imsave("./multi" + str(i) + ".png", rendering)

    np.save("all_parameters", allParams)
    plt.show()