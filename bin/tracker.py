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
    all_param = np.load("./parameters.npy")
    texCoef = all_param[:m.numTex]
    shCoef = all_param[m.numTex: m.numTex + 27].reshape(9, 3)
    param = all_param[m.numTex + 27:]
    idCoef = param[:m.numId]
    expCoef = param[m.numId : m.numId + m.numExp]


    for frame in np.arange(startFrame, numFrames):
        print(frame)
        fName = 'subject_1_{:0>12}'.format(frame)
        
        """
        Set filenames, read landmarks, load source video frames
        """
        # Frames from the source video
        fNameImgOrig = 'orig/' + fName + '_rendered.png'

        # Load the source video frame and convert to 64-bit float
        img_org = io.imread(fNameImgOrig)
        img = img_as_float(img_org)

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


        # """
        # Get initial spherical harmonic lighting parameter guess
        # """
        # # Calculate normals at each vertex in the 3DMM
        # vertexNorms = calcNormals(vertexCoords, m)

        # # Evaluate spherical harmonics at face shape normals. The result is a (numVertices, 9) array where each column is a spherical harmonic basis for the 3DMM.
        # B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        
        # # Get the pixel RGB values of the original image where the 3DMM face is rendered
        # imgMasked = img[pixelCoord[:, 0], pixelCoord[:, 1]]
        
        # # Initialize an array to store the barycentric reconstruction of the nine spherical harmonics. The first dimension indicates the color (RGB).
        # I = np.empty((3, pixelFaces.size, 9))
        
        # # Loop through each color channel
        # for c in range(3):
        #     # 
        #     I[c, ...] = barycentricReconstruction(B * texture[c, :], pixelFaces, pixelBarycentricCoords, m.face)

        #     # Make an initial guess of the spherical harmonic lighting coefficients with least squares. We are solving Ax = b, where A is the (numFaces, 9) array of the barycentric reconstruction of the spherical harmonic bases, x is the (9,) vector of coefficients, and b is the (numFaces,) vector of the pixels from the original image where the 3DMM is defined.
        #     shCoef[:, c] = lsq_linear(I[c, ...], imgMasked[:, c]).x


        # """
        # Optimization over all experssion
        # """
        # numRandomFaces =  10000
        # randomFaces = np.random.randint(0, pixelFaces.size - 300, numRandomFaces)
        initFit = least_squares(opt.denseExpResiduals, param[m.numId:], jac = opt.denseExpJacobian, args = (idCoef, texCoef, shCoef, lm, img, m, renderObj, (1, 4, 0.004)), loss = 'linear', verbose = 0, max_nfev = 10)
        param = np.r_[idCoef, initFit['x']]
        expCoef = param[m.numId: m.numId + m.numExp]

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

        # plt.figure("Dense Shape 2")
        # plt.imshow(rendering)
        scipy.misc.imsave("./" + str(frame) + ".png", rendering)


        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting 2")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()

