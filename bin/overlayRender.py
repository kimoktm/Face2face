#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.utils.mesh import generateFace, generateTexture, barycentricReconstruction, writePly

import cv2
import scipy.misc
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.transform import resize
import matplotlib.pyplot as plt

import os
import glob
import argparse


def saveImage(path, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = img_as_ubyte(img)
    cv2.imwrite(path, img)

def main():
    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('./data')

    # Load 3DMM
    m = MeshModel('../models/bfm2017.npz')
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'

    # load texture if set
    vertexImgColor = None
    if FLAGS.img_texture is not None:
        vertexImgColor = np.load(os.path.join(FLAGS.img_texture))

    # apply mask on faces if supplied
    if FLAGS.face_mask is not None:
        mask_id = np.load(FLAGS.face_mask)
        m.face = np.delete(m.face, mask_id, axis = 0)
        m.vertex2face = np.array([np.where(np.isin(m.face.T, vertexInd).any(axis = 0))[0] for vertexInd in range(m.numVertices)])

    data_path = os.path.join(FLAGS.input_dir, '*.png')
    keyframes = glob.glob(data_path)

    for i in range(FLAGS.start_frame, FLAGS.start_frame + 50):
        print(i)
        fNameImgOrig = os.path.join(FLAGS.input_dir, str(i) + '.png')

        # Load the source video frame and convert to 64-bit float
        b,g,r = cv2.split(cv2.imread(fNameImgOrig))
        img_org = cv2.merge([r,g,b])
        img_org = cv2.GaussianBlur(img_org, (3, 3), 0)
        img = img_as_float(img_org)

        # """
        # Rendering
        # """
        # load parameters
        frame_params = np.load(os.path.join(FLAGS.params_dir, str(i) + "_params.npy"))
        shCoef = frame_params[:27]
        param = frame_params[27:]

        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # Generate the texture at the 3DMM vertices from the learned texture coefficients
        texParam = np.r_[np.zeros((m.numTex)), shCoef.flatten()]
        vertexImgColor = vertexImgColor*0 +0.7
        texture = generateTexture(vertexCoords, texParam, m, vertexImgColor)

        # Render the 3DMM
        renderObj = Render(img.shape[1], img.shape[0], np.r_[vertexCoords.T, texture.T], m.face, False, img)
        renderObj.render()
        # rendering = renderObj.grabRendering()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        renderObj.closeRender()

        # # CPU rendering
        # Using the barycentric parameters from the rendering, we can reconstruct the image with the 3DMM texture model by taking barycentric combinations of the 3DMM RGB values defined at the vertices
        # imgReconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, m.face)
        # # Put values from the reconstruction into a (height, width, 3) array for plotting
        # reconstruction = img
        # reconstruction[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
        # # print(img[pixelCoord[:, 0], pixelCoord[:, 1]])
        # # print(imgReconstruction)

        saveImage(os.path.join(FLAGS.output_dir, str(i) + ".png"), rendering)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Render fitted parameters')
    parser.add_argument('--input_dir', help = 'Path to images')
    parser.add_argument('--params_dir', help = 'Path to fitted expression & light parameters')
    parser.add_argument('--output_dir', help = 'Output directory')
    parser.add_argument('--img_texture', help = 'Path to texture (vertex space) instead of PCA model (optional)')
    parser.add_argument('--face_mask', help = 'Path to face ids to mask as eyes (optional)')
    parser.add_argument('--start_frame', help = 'Frame to start tracking from (optional)',type = int, default = 0)

    FLAGS, unparsed = parser.parse_known_args()

    main()