#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estimateCamMat, splitCamMat
import mm.optimize.image as opt
from mm.utils.mesh import generateFace, generateTexture, writePly

import dlib
import cv2
import scipy.misc
import numpy as np
from scipy.optimize import least_squares
from skimage import io, img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt

import os
import glob
import argparse
import time



def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=320):
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


def main():
    # Set weights for the 3DMM RGB color shape, landmark shape, and regularization terms
    max_iterations = 10
    wCol = 1
    # wLan = 2.5e-5
    # wRegS = 1.25e-4
    wLan = 1.25e-5
    wRegS = 0.25e-4
    # lsmr is numerically accurate but slower
    tr_solver = 'lsmr'


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
    
    # Load parameters
    all_param = np.load(FLAGS.parameters)
    texCoef = all_param[:m.numTex]
    shCoef = all_param[m.numTex: m.numTex + 27]
    param = all_param[m.numTex + 27:]
    idCoef = param[:m.numId]
    expCoef = param[m.numId : m.numId + m.numExp]

    data_path = os.path.join(FLAGS.input_dir, '*.png')
    keyframes = glob.glob(data_path)

    for i in range(FLAGS.start_frame, len(keyframes)):
        print(i)
        fNameImgOrig = os.path.join(FLAGS.input_dir, str(i) + '.png')

        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(fNameImgOrig))
        img_org = cv2.merge([r,g,b])
        img_org = cv2.GaussianBlur(img_org, (3, 3), 0)
        img = img_as_float(img_org)

        # plt.figure("Blurre")
        # plt.imshow(img)
        # plt.show()

        shape2D = getFaceKeypoints(img_org, detector, predictor)
        shape2D = np.asarray(shape2D)[0].T 
        lm = shape2D[m.targetLMInd, :2]

        if i == FLAGS.start_frame:
            vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
            # Rendering of initial 3DMM shape with mean texture model
            texParam = np.r_[texCoef, shCoef.flatten()]
            texture = generateTexture(vertexCoords, texParam, m)
            meshData = np.r_[vertexCoords.T, texture.T]
            renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)
            renderObj.render()

            # Grab the OpenGL rendering from the video card
            rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
            # scipy.misc.imsave(os.path.join(FLAGS.output_dir, str(i) + "_orig.png"), rendering)
            # plt.figure("Initial")
            # plt.imshow(rendering)

            # Adjust Landmarks to be consistent across segments
            p1_id = 27 # nose
            p2_id = 8  # jaw
            x2 = lm[p1_id, 0]
            x1 = lm[p2_id, 0]
            y2 = lm[p1_id, 1]
            y1 = lm[p2_id, 1]
            nosejaw_dist = ((x2 - x1)**2 + (y2 - y1)**2)**(1/2)
            wLan = wLan * (225.0 / nosejaw_dist)


        # """
        # Optimization over all experssion & SH
        # """
        # LSMR is numerically stable combared to the default option (Exact)
        initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], tr_solver = tr_solver, max_nfev = max_iterations, jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img, lm, m, renderObj, (wCol, wLan, wRegS)), verbose = 0, x_scale = 'jac')
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

        scipy.misc.imsave(os.path.join(FLAGS.output_dir, str(i) + ".png"), rendering)
        np.save(os.path.join(FLAGS.output_dir, str(i) + "_params"), np.r_[shCoef, param])

        # plt.figure("Dense Shape 3")
        # plt.imshow(rendering)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting 3")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Initialize Identity & Texture from multiple frames')
    parser.add_argument('--input_dir', help = 'Path to frames')
    parser.add_argument('--parameters', help = 'Path to parameters to start tracking')
    parser.add_argument('--start_frame', help = 'Frame to start tracking from',type = int, default = 0)
    parser.add_argument('--output_dir', help = 'Output directory')

    FLAGS, unparsed = parser.parse_known_args()

    main()