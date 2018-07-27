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
from skimage import io, img_as_float, img_as_ubyte
from skimage.transform import resize
import matplotlib.pyplot as plt

import os
import glob
import argparse
import time
import pandas as pd
from tqdm import tqdm



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
    max_iterations = 9
    wCol = 1
    # old
    # wLan = 2.5e-5
    # wRegS = 1.25e-4
    
    # init
    # wLan = 2.9e-5
    # wRegS = 0.25e-5

    # dlib
    # wLan = 1.25e-5
    # wRegS = 0.25e-5
    wLan = 0
    wRegS = 2.5e-5

    # openFace - Test
    # wLan = 1.3e-5
    # wRegS = 0.6e-4

    # lsmr is numerically stable and faster
    tr_solver = 'lsmr'

    # corase-to-fine iteration scheme
    ctf_iterations = [5, 3, 2]
    ctf_levels     = [4, 2, 1]
    ctf_renderObjs = []


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

    # apply mask on faces if supplied
    if FLAGS.face_mask is not None:
        mask_id = np.load(FLAGS.face_mask)
        m.face = np.delete(m.face, mask_id, axis = 0)
        m.vertex2face = np.array([np.where(np.isin(m.face.T, vertexInd).any(axis = 0))[0] for vertexInd in range(m.numVertices)])

    # Load parameters
    all_param = np.load(FLAGS.parameters)
    texCoef = all_param[:m.numTex]
    shCoef = all_param[m.numTex: m.numTex + 27]
    param = all_param[m.numTex + 27:]
    idCoef = param[:m.numId]
    expCoef = param[m.numId : m.numId + m.numExp]

    vertexImgColor = None
    if FLAGS.img_texture is not None:
        vertexImgColor = np.load(os.path.join(FLAGS.img_texture))

    data_path = os.path.join(FLAGS.input_dir, '*.png')
    keyframes = glob.glob(data_path)

    start = time.time()

    for i in tqdm(range(FLAGS.start_frame, len(keyframes))):
        fNameImgOrig = os.path.join(FLAGS.input_dir, str(i) + '.png')

        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(fNameImgOrig))
        img_org = cv2.merge([r,g,b])
        img_org = cv2.GaussianBlur(img_org, (5, 5), 0)
        img = img_as_float(img_org)

        if FLAGS.openFace_landmarks is None:
            shape2D = getFaceKeypoints(img_org, detector, predictor)
            shape2D = np.asarray(shape2D)[0].T
        else:
            shape2D = loadOpenFaceKeypoints(i + 1, openFaceData)
            shape2D = np.asarray(shape2D)

        lm = shape2D[m.targetLMInd, :2]

        if i == FLAGS.start_frame:
            vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
            # Rendering of initial 3DMM shape with mean texture model
            texParam = np.r_[texCoef, shCoef.flatten()]
            meshData = np.r_[vertexCoords.T, m.texMean.T]

            # create renderObjs for corase to fine
            for level in ctf_levels:
                ctf_renderObjs.append(Render(int(img.shape[1] / level), int(img.shape[0] / level), meshData, m.face))

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
        # Coarse-to-Fine Optimization over all experssion & SH
        # """
        for l, level in enumerate(ctf_levels):
            img_resized = cv2.resize(img, (int(img.shape[1] / level), int(img.shape[0] / level)))
            # img_resized = resize(img, (int(img.shape[0] / level), int(img.shape[1] / level)))
            param[-3:] = param[-3:] / level
            initFit = least_squares(opt.denseJointExpResiduals, np.r_[shCoef, param[m.numId:]], tr_solver = tr_solver, max_nfev = ctf_iterations[l], jac = opt.denseJointExpJacobian, args = (idCoef, texCoef, img_resized, lm / level, m, ctf_renderObjs[l], (wCol, wLan * level, wRegS), vertexImgColor), verbose = 0, x_scale = 'jac')
            shCoef = initFit['x'][:27]
            expCoef = initFit['x'][27:]
            param = np.r_[idCoef, expCoef]
            param[-3:] = param[-3:] * level

        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # Generate the texture at the 3DMM vertices from the learned texture coefficients
        texParam = np.r_[texCoef, shCoef.flatten()]
        texture = generateTexture(vertexCoords, texParam, m, vertexImgColor)

        # Render the 3DMM
        ctf_renderObjs[-1].updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        ctf_renderObjs[-1].resetFramebufferObject()
        ctf_renderObjs[-1].render()
        rendering = ctf_renderObjs[-1].grabRendering()

        saveImage(os.path.join(FLAGS.output_dir, str(i) + ".png"), rendering)
        np.save(os.path.join(FLAGS.output_dir, str(i) + "_params"), np.r_[shCoef, param])

    elapsed = time.time() - start
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Initialize Identity & Texture from multiple frames')
    parser.add_argument('--input_dir', help = 'Path to frames')
    parser.add_argument('--parameters', help = 'Path to parameters to start tracking')
    parser.add_argument('--output_dir', help = 'Output directory')
    parser.add_argument('--openFace_landmarks', help = 'Path to openface landmarks otherwise dlib will be used (optional)')
    parser.add_argument('--img_texture', help = 'Path to texture (vertex space) instead of PCA model (optional)')
    parser.add_argument('--face_mask', help = 'Path to face ids to mask as eyes (optional)')
    parser.add_argument('--start_frame', help = 'Frame to start tracking from',type = int, default = 0)

    FLAGS, unparsed = parser.parse_known_args()

    main()