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

# delete
from autograd import value_and_grad, grad, jacobian, hessian
import scipy.misc


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
        # writePly("../mesh.ply", vertexCoords, m.face, m.texMean, m.sourceLMInd)

        # Estimate the camera projection matrix from the landmark correspondences
        camMat = estimateCamMat(lm, lm3D, cam)
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(camMat, cam)
        
        # Concatenate parameters for input into optimization routine. Note that the translation vector here is only (2,) for x and y (no z)
        param = np.r_[idCoef, expCoef, angles, t, s]


        # # TEST JACOBIANS
        # print(jacobian(opt.initialShapeCost)(param, lm, m, (wLan, 0)))
        # print("####################################")
        # print(opt.initialShapeGrad(param, lm, m, (wLan, 0)))
        # break

        start = time.time()

        # Initial optimization of shape parameters with similarity transform parameters
        # initFit = minimize(value_and_grad(opt.initialShapeCost), param, args = (lm, m, (wLan, 0.05)), method='BFGS', jac=True, options={'disp': True})
        # param = initFit.x

        # initFit = minimize(value_and_grad(opt.initialShapeCost), param, jac = True, hess = hessian(opt.initialShapeCost), args = (lm, m, (wLan, 0.05)), method='trust-exact', options={'disp': True})
        # param = initFit.x

        # initFit = least_squares(opt.initialShapeCost, param, jac=jacobian(opt.initialShapeCost), args = (lm, m, (wLan, wReg)), tr_solver = 'lsmr', loss = 'linear',verbose = 2)
        # param = initFit['x']

        # initFit = least_squares(opt.initialShapeResiuals, param, jac=jacobian(opt.initialShapeResiuals), args = (lm, m, (wLan, wReg)),loss = 'linear', verbose = 2)
        # param = initFit['x']

        # Initial optimization of shape parameters with similarity transform parameters
        # initFit = least_squares(opt.initialShapeResiuals, param, jac = opt.initialShapeJacobians, args = (lm, m, (wLan, wReg)), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
        # param = initFit['x']

        elapsed = time.time() - start
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        # break

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
        # plt.show()
        # break

        # # CPU rendering
        # # Using the barycentric parameters from the rendering, we can reconstruct the image with the 3DMM texture model by taking barycentric combinations of the 3DMM RGB values defined at the vertices
        # imgReconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, m.face)
        
        # # Put values from the reconstruction into a (height, width, 3) array for plotting
        # reconstruction = np.zeros(rendering.shape)
        # reconstruction[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction

        # print(rendering.shape)
        # print(imgReconstruction[:,0])
        # print(reconstruction[:10,0])
        # print("#######################")


        # # Plot the difference of the reconstruction with the rendering to see that they are very close-- the output values should be close to 0
        # plt.figure()
        # plt.imshow(np.fabs(reconstruction - rendering))
        # plt.show()


        # # """
        # # Get initial texture parameter guess
        # # """
        
        # # # Set the number of faces for stochastic optimization
        # numRandomFaces = 10000

        # wReg = 0.00005

        # # # TEST JACOBIANS
        # # print(jacobian(opt.textureCost)(texCoef, img, vertexCoords, m, renderObj, (wCol, wReg)))
        # # print("####################################")
        # # print(opt.textureGrad(texCoef, img, vertexCoords, m, renderObj, (wCol, wReg)))
        # # break

        # # Do some cycles of nonlinear least squares iterations, using a new set of random faces each time for the optimization objective
        # for i in range(1):
        #     randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     initTex = least_squares(opt.textureResiduals, texCoef, jac = opt.textureJacobian, args = (img, vertexCoords, m, renderObj, (wCol, wReg), randomFaces), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
        #     texCoef = initTex['x']

        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        
        # # Update the rendering and plot
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # # # writePly("../mesh_landmarks.ply", vertexCoords, m.face, texture)

        # plt.figure("Texture")
        # plt.imshow(rendering)
        # # plt.show()


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

        # wReg = 0.0000005
        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # for i in range(1):
        #     # randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, vertexCoords, B, m, renderObj, (wCol, wReg)), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
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


        # # """
        # # Optimization over dense shape
        # # """
        # # # TEST JACOBIANS
        # # print(jacobian(opt.denseCost)(param, img, texCoef, m, renderObj, vertexCoords, (wLan, wReg)))
        # # print("####################################")
        # # print(opt.denseGrad(param, img, texCoef, m, renderObj, (wLan, wReg)))
        # # break

        # # # SHIFTED
        # # param[-3] = param[-3] + 5.0
        # # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # # renderObj.resetFramebufferObject()
        # # renderObj.render()
        # # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        # # plt.figure("Shifted")
        # # plt.imshow(rendering)
        # # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # # plt.figure("Shifted fitting")
        # # plt.imshow(img)
        # # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # # # plt.show()

        # initFit = least_squares(opt.denseResiduals, param, jac = opt.denseJacobian, args = (img, texCoef, m, renderObj, (wLan, 0.000005)), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
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
        # break


        # # """
        # # Optimization over dense shape & albedo
        # # """
        # param = np.r_[texCoef, param]

        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # initFit = least_squares(opt.denseTexResiduals, param, jac = opt.denseTexJacobian, args = (img, m, renderObj, (wLan, 0.00005, 0.000001)), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
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


        # # """
        # # Optimization over dense shape, Albedo & SH
        # # """
        # shCoef = np.zeros((9, 3))
        # shCoef[0, 0] = 0.5
        # shCoef[0, 1] = 0.5
        # shCoef[0, 2] = 0.5
        # allParam = np.r_[texCoef, shCoef.flatten(), param]

        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # initFit = least_squares(opt.denseAllResiduals, allParam, jac = opt.denseAllJacobian, args = (img, m, renderObj, (1, 0.00000005, 0.000000005)), max_nfev = 10, loss = 'linear', method = 'trf', tr_solver='lsmr', verbose = 2)

        # allParam = initFit['x']
        # texParam3 = allParam[:texCoef.size + shCoef.size]
        # shapeParam3 = allParam[texCoef.size + shCoef.size:]

        # # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[shapeParam3[:-1], 0, shapeParam3[-1]], m)

        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texture = generateTexture(vertexCoords, texParam3, m)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # plt.figure("Dense Shape 1")
        # plt.imshow(rendering)

        # writePly("../mesh_step1.ply", vertexCoords, m.face, texture)

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()



        # # """
        # # Optimization over all dense shape with SH fixed
        # # """
        # param = np.r_[texCoef, param]

        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # initFit = least_squares(opt.denseShResiduals, param, jac = opt.denseShJacobian, args = (img, shCoef.flatten(), lm, m, renderObj, (1, 6, 0.0002, 0.08)), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
        # param_all = initFit['x']
        # texCoef = param_all[:m.numTex]
        # param = param_all[m.numTex:]

        # # Generate 3DMM vertices from shape and similarity transform parameters
        # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)

        # # Generate the texture at the 3DMM vertices from the learned texture coefficients
        # texParam = np.r_[texCoef, shCoef.flatten()]
        # texture = generateTexture(vertexCoords, texParam, m)

        # # Render the 3DMM
        # renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        # renderObj.resetFramebufferObject()
        # renderObj.render()
        # rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)

        # plt.figure("Dense Shape 2")
        # plt.imshow(rendering)

        # writePly("../mesh_step2.ply", vertexCoords, m.face, texture)
        # np.save("parameters", np.r_[texCoef, shCoef.flatten(), param])

        # # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        # plt.figure("Desne fitting 2")
        # plt.imshow(img)
        # plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        # plt.show()


        """
        Optimization over shape, texture, and lighting
        """
        shCoef = np.zeros((9, 3))
        shCoef[0, 0] = 0.5
        shCoef[0, 1] = 0.5
        shCoef[0, 2] = 0.5
        allParam = np.r_[texCoef, shCoef.flatten(), param]

        # # TEST JACOBIANS
        # vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        # print(jacobian(opt.denseJointCost)(allParam, img, lm, m, renderObj, vertexCoords, (1, 5, 0.0002, 0.09)))
        # print("####################################")
        # print(opt.denseJointGrad(allParam, img, lm, m, renderObj, (1, 0, 0.0, 0.0)))
        # break

        # Jointly optimize the texture and spherical harmonic lighting coefficients
        initShapeTexLight = least_squares(opt.denseJointResiduals, allParam, jac = opt.denseJointJacobian, args = (img, lm, m, renderObj, (1, 1, 0.00000000005, 0.008)), loss = 'linear', verbose = 2, max_nfev = 50, method = 'trf', tr_solver='lsmr')
        allParam = initShapeTexLight['x']

        # initFit = minimize(opt.denseJointCost, allParam, jac = opt.denseJointGrad, args = (img, lm, m, renderObj, (1, 5, 0.000002, 0.08)), method='BFGS', options={'disp': True, 'maxiter': 10})
        # allParam = initFit.x

        texParam3 = allParam[:texCoef.size + shCoef.size]
        shapeParam3 = allParam[texCoef.size + shCoef.size:]

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




        # """
        # Optimization simultaneously over lighting parameters
        # """
        # shCoef = np.ones((9, 3)) * 0.0
        # shCoef[0, 0] = 0.5
        # shCoef[0, 1] = 0.5
        # shCoef[0, 2] = 0.5
        # # print(shCoef)
        # # Calculate normals at each vertex in the 3DMM
        # vertexNorms = calcNormals(vertexCoords, m)

        # # Evaluate spherical harmonics at face shape normals. The result is a (numVertices, 9) array where each column is a spherical harmonic basis for the 3DMM.
        # B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        # texParam2 = shCoef.flatten()


        # # print(jacobian(opt.lightingCost)(texParam2, texCoef, img, vertexCoords, B, m, renderObj).reshape(9, 3))
        # # print("####################################")
        # # print(opt.lightingGrad(texParam2, texCoef, img, vertexCoords, B, m, renderObj).reshape(9, 3))
        # # break

        # wReg = 0
        # # Jointly optimize the texture and spherical harmonic lighting coefficients
        # for i in range(1):
        #     # randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
        #     # initFit = minimize(value_and_grad(opt.lightingResiduals), texParam2, jac = True, args = (texCoef, img, vertexCoords, B, m, renderObj), options={'disp': True}, tol = 0.001, method = 'BFGS')
        #     # texParam2 = initFit.x

        #     initTexLight = least_squares(opt.lightingResiduals, texParam2, jac = opt.lightingJacobian, args = (texCoef, img, vertexCoords, B, m, renderObj), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2)
        #     # initTexLight = least_squares(opt.lightingResiduals, texParam2, jac = opt.lightingGrad, args = (texCoef, img, vertexCoords, B, m, renderObj), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2, max_nfev = 20)
        #     # initTexLight = least_squares(opt.lightingResiduals, texParam2, jac = jacobian(opt.lightingResiduals), args = (texCoef, img, vertexCoords, B, m, renderObj), method = 'trf', tr_solver='lsmr', loss = 'linear', verbose = 2,  max_nfev = 10)
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