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
    wLan = 1.25e-4
    wRegC = 2.5e-5
    wRegS = 1.25e-5

    for frame in np.arange(0, 1):
        print(frame)

        fName = '{:0>6}'.format(frame)
        fNameImgOrig = 'obama/orig/' + fName + '.png'

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
        scale_factor = 1.0
        img = resize(img, (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)))
        lm = lm / scale_factor


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
        # writePly("../mesh.ply", vertexCoords, m.face, m.texMean, m.sourceLMInd)

        # Estimate the camera projection matrix from the landmark correspondences
        camMat = estimateCamMat(lm, lm3D, cam)
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(camMat, cam)
        
        # Concatenate parameters for input into optimization routine. Note that the translation vector here is only (2,) for x and y (no z)
        param = np.r_[idCoef, expCoef, angles, t, s]



        """
        Optimization over shape, texture, and lighting
        """
        shCoef = np.zeros((9, 3))
        shCoef[0, 0] = 0.5
        shCoef[0, 1] = 0.5
        shCoef[0, 2] = 0.5
        allParam = np.r_[texCoef, shCoef.flatten(), param]

        # Rendering of initial 3DMM shape with mean texture model
        texture = m.texMean
        meshData = np.r_[vertexCoords.T, texture.T]
        renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)



        # Jointly optimize the texture and spherical harmonic lighting coefficients
        start = time.time()

        initShapeTexLight = least_squares(opt.denseJointResiduals, allParam, jac = opt.denseJointJacobian, tr_solver = 'lsmr', max_nfev = 35, args = (img, lm, m, renderObj, (wCol, wLan, wRegC, wRegS)), verbose = 2, x_scale = 'jac')
        allParam = initShapeTexLight['x']
        texParam3 = allParam[:texCoef.size + shCoef.size]
        shapeParam3 = allParam[texCoef.size + shCoef.size:]

        elapsed = time.time() - start
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))



        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[shapeParam3[:-1], 0, shapeParam3[-1]], m)

        # Generate 3DMM texture form vertex & sh parameters
        texture = generateTexture(vertexCoords, texParam3, m)

        # Render the 3DMM
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        print(texParam3[texCoef.size:].reshape(9, 3))

        plt.figure("Shape & Texture & SH")
        plt.imshow(rendering)

        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        plt.figure("Desne fitting")
        plt.imshow(img)
        plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        
        writePly("../mesh_sh_all.ply", vertexCoords, m.face, texture)
        scipy.misc.imsave("./" + str(frame) + ".png", rendering)
        np.save("parameters", allParam)
        plt.show()