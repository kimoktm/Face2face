#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estimateCamMat, splitCamMat
import mm.optimize.image as opt
from mm.utils.mesh import generateFace, generateTexture, getImgsColors, writePly

import dlib
import cv2
import scipy.misc
import numpy as np
from scipy.optimize import least_squares
from skimage import io, img_as_float, img_as_ubyte
from skimage.transform import resize
import matplotlib.pyplot as plt

import os
import glob
import argparse
import time
import pandas as pd



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


def loadOpenFaceKeypoints(frame_cnt, openFace_landmarks):
    shapes2D = []
    frame = openFace_landmarks[openFace_landmarks['frame'] == frame_cnt]

    for i in range(0, 68):
        x = frame[' x_' + str(i)].values[0]
        y = frame[' y_' + str(i)].values[0]
        shapes2D.append([x, y])

    return shapes2D


def saveImage(path, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = img_as_ubyte(img)
    cv2.imwrite(path, img)


def main():
    # Set weights for the 3DMM RGB color shape, landmark shape, and regularization terms
    scale_factor = 1.0
    max_iterations = 32
    wCol = 1
    # wLan = 1.25e-4
    wLan = 2.9e-5
    wRegC = 0.025e-5
    wRegS = 0.25e-5
    # lsmr is numerically stable and faster
    tr_solver = 'lsmr'


    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('./data')

    # Load 3DMM
    m = MeshModel('../models/bfm2017.npz')
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'

    # Landmark detector
    if FLAGS.openFace_landmarks is None:
        print('Using dlib landmarks...')
        predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
    else:
        print('Using openFace landmarks...')
        openFaceData = pd.read_csv(FLAGS.openFace_landmarks)

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

    # load images & estimate camera params
    data_path = os.path.join(FLAGS.input_dir, '*g')
    keyframes = glob.glob(data_path)
    keyframes.sort()

    for i, frame in enumerate(keyframes):
        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(frame))
        img_org = cv2.merge([r,g,b])
        #img_org = cv2.GaussianBlur(img_org, (3, 3), 0)
        img = img_as_float(img_org)

        if FLAGS.openFace_landmarks is None:
            shape2D = getFaceKeypoints(img_org, detector, predictor)
            shape2D = np.asarray(shape2D)[0].T
        else:
            frame_name = os.path.splitext(os.path.basename(frame))[0]
            shape2D = loadOpenFaceKeypoints(int(frame_name) + 1, openFaceData)
            shape2D = np.asarray(shape2D)

        lm = shape2D[m.targetLMInd, :2]

        # Resize image for speed
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

        if i == 0:
            # Initialize render Object
            texture = m.texMean
            meshData = np.r_[vertexCoords.T, texture.T]
            renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)

            # Adjust Landmarks to be consistent across segments
            p1_id = 27 # nose
            p2_id = 8  # jaw
            x2 = lm[p1_id, 0]
            x1 = lm[p2_id, 0]
            y2 = lm[p1_id, 1]
            y1 = lm[p2_id, 1]
            nosejaw_dist = ((x2 - x1)**2 + (y2 - y1)**2)**(1/2)
            wLan = wLan * (225.0 / nosejaw_dist)


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
    initShapeTexLight = least_squares(opt.multiDenseJointResiduals, allParams, jac = opt.multiDenseJointJacobian, tr_solver = tr_solver, max_nfev = max_iterations, args = (imgs, landmarks, m, renderObj, (wCol, wLan, wRegC, wRegS)), verbose = 2, x_scale = 'jac')
    allParams = initShapeTexLight['x']
    texCoef = allParams[: texCoef.size]
    idCoef = allParams[texCoef.size : texCoef.size + idCoef.size]
    img_params = allParams[texCoef.size + idCoef.size :].reshape(img_params.shape)

    elapsed = time.time() - start
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    # Visualize results
    shCoefList = []
    vertexCoordsList = []
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

        # # print(texParam[texCoef.size:].reshape(9, 3))
        # plt.figure(str(i) + " Shape & Texture & SH")
        # plt.imshow(rendering)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure(str(i) + " Desne fitting")
        # plt.imshow(imgs[i])
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(landmarks[i, :, 0], landmarks[i, :, 1], s = 2, c = 'g')

        saveImage(os.path.join(FLAGS.output_dir, "multi" + str(i) + ".png"), rendering)
        vertexCoordsList.append(vertexCoords)
        shCoefList.append(shCoef.reshape((9, 3)))

        if i == 0:
            expCoef[-3:] * scale_factor
            first_frame_param = np.r_[texCoef, shCoef, idCoef, expCoef]
            np.save(os.path.join(FLAGS.output_dir, "params"), first_frame_param)


    # Texture mesh
    vertexImgColor = getImgsColors(vertexCoordsList, shCoefList, imgs, m, renderObj)
    np.save(os.path.join(FLAGS.output_dir, "texture"), vertexImgColor)

    for i in range(img_params.shape[0]):
        lightedImgColor = generateTexture(vertexCoordsList[i], np.r_[texCoef, shCoefList[i].flatten()], m, vertexImgColor)
        renderObj = Render(imgs[i].shape[1], imgs[i].shape[0], np.r_[vertexCoordsList[i].T, lightedImgColor.T], m.face, False, imgs[i])
        renderObj.render()
        rendering = renderObj.grabRendering()
        saveImage(os.path.join(FLAGS.output_dir, "textured_" + str(i) + ".png"), rendering)

        if i == 0:
            writePly(os.path.join(FLAGS.output_dir, "textured_mesh.ply"), vertexCoordsList[0], m.face, lightedImgColor)


    # TO DO: scale poses up
    np.save(os.path.join(FLAGS.output_dir, "all_params"), allParams)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Initialize Identity & Texture from multiple frames')
    parser.add_argument('--input_dir', help = 'Path to frames')
    parser.add_argument('--output_dir', help = 'Output directory')
    parser.add_argument('--openFace_landmarks', help = 'Path to openface landmarks otherwise dlib will be used (optional)')

    FLAGS, unparsed = parser.parse_known_args()

    main()