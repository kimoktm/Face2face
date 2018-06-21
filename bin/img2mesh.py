#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estimateCamMat, splitCamMat
import mm.optimize.image as opt
from mm.utils.mesh import calcNormals, generateFace, generateTexture, barycentricReconstruction
from mm.utils.transform import sh9

import os, json
import numpy as np
from scipy.optimize import minimize, check_grad, least_squares, nnls, lsq_linear
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import savefig

if __name__ == "__main__":
    
    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('/home/karim/Desktop/face-fitting/data')
    
    # Input the number of frames in the video
    numFrames = 995 #2260 #3744
    
    # Load 3DMM
    m = MeshModel('/home/karim/Desktop/face-fitting/models/bfm2017.npz')
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'
    
    # Set weights for the 3DMM RGB color shape, landmark shape, and regularization terms
    wCol = 1000
    wLan = 10
    wReg = 1

    # # Data for plotting
    # t = np.arange(0.0, 2.0, 0.01)
    # s = 1 + np.sin(2 * np.pi * t)

    # # Note that using plt.subplots below is equivalent to using
    # # fig = plt.figure() and then ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    # ax.plot(t, s)

    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    # ax.grid()

    # fig.savefig("test.png")
    # plt.show()
    
    for frame in np.arange(0, 1):
        print(frame)
        fName = 'subject_1_{:0>12}'.format(frame)
        
        """
        Set filenames, read landmarks, load source video frames
        """
        # Frames from the source video
        fNameImgOrig = 'orig/' + fName + '_rendered.png'
        
        # OpenPose landmarks for each frame in the source video
        fNameLandmarks = 'landmark/' + fName + '_keypoints.json'
        
        with open(fNameLandmarks, 'r') as fd:
            lm = json.load(fd)

        lm = np.array(lm['people'][0]['face_keypoints_2d']).reshape([-1, 3])
        # lm = np.array([l[0] for l in lm], dtype = int).squeeze()[:, :3]
        lmConf = lm[m.targetLMInd, -1]  # This is the confidence value of the landmarks
        lm = lm[m.targetLMInd, :2]

        # Load the source video frame and convert to 64-bit float
        img = io.imread(fNameImgOrig)
        img = img_as_float(img)
        

        imgv = mpimg.imread(fNameImgOrig)
        imgplot = plt.imshow(imgv)
        
        # You can plot the landmarks over the frames if you want
        # plt.figure()
        # plt.imshow(img)
        # plt.scatter(lm[:, 0], lm[:, 1], s = 2)
        # plt.title(fName)
        # if not os.path.exists('landmarkPic'):
        #    os.makedirs('landmarkPic')
        # savefig('../landmarkPic/' + fName + '.png', bbox_inches='tight')
        # plt.close('all')
        # plt.close()

        # fig, ax = plt.subplots()
        # plt.imshow(img)
        # plt.hold(True)
        # x = lm[:, 0]
        # y = lm[:, 1]
        # ax.scatter(x, y, s = 2, c = 'b', picker = True)
        # fig.canvas.mpl_connect('pick_event', onpick3)
        
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
        
        # Estimate the camera projection matrix from the landmark correspondences
        camMat = estimateCamMat(lm, lm3D, cam)
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(camMat, cam)
        
        # Concatenate parameters for input into optimization routine. Note that the translation vector here is only (2,) for x and y (no z)
        param = np.r_[idCoef, expCoef, angles, t, s]
        
        # Initial optimization of shape parameters with similarity transform parameters
        initFit = minimize(opt.initialShapeCost, param, args = (lm, m, (wLan, wReg)), jac = opt.initialShapeGrad)
        param = initFit.x
        idCoef = param[:m.numId]
        expCoef = param[m.numId: m.numId+m.numExp]
        
        # Generate 3DMM vertices from shape and similarity transform parameters
        vertexCoords = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        
        # Plot the 3DMM in 3D
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(vertexCoords[0, :], vertexCoords[1, :], vertexCoords[2, :])
        
        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
        plt.figure()
        plt.imshow(img)
        plt.scatter(vertexCoords[0, m.sourceLMInd], vertexCoords[1, m.sourceLMInd], s = 3, c = 'r')
        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'g')
        
        # Rendering of initial 3DMM shape with mean texture model
        texture = m.texMean
        meshData = np.r_[vertexCoords.T, texture.T]
        renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)
        renderObj.render()
        
        # Grab the OpenGL rendering from the video card
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        # Plot the OpenGL rendering
        plt.figure()
        plt.imshow(rendering)
        
        # Using the barycentric parameters from the rendering, we can reconstruct the image with the 3DMM texture model by taking barycentric combinations of the 3DMM RGB values defined at the vertices
        imgReconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, m.face)
        
        # Put values from the reconstruction into a (height, width, 3) array for plotting
        reconstruction = np.zeros(rendering.shape)
        reconstruction[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
        
        # Plot the difference of the reconstruction with the rendering to see that they are very close-- the output values should be close to 0
        plt.figure("difference")
        plt.imshow(np.fabs(reconstruction - rendering))
        
        """
        Get initial texture parameter guess
        """
        
        # Set the number of faces for stochastic optimization
        numRandomFaces = 10000
        
        # Do some cycles of nonlinear least squares iterations, using a new set of random faces each time for the optimization objective
        cost = np.zeros(20)
        for i in range(20):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTex = least_squares(opt.textureResiduals, texCoef, jac = opt.textureJacobian, args = (img, vertexCoords, m, renderObj, (wCol, wReg), randomFaces), loss = 'soft_l1')
            texCoef = initTex['x']
            cost[i] = initTex.cost
        
        # Generate the texture at the 3DMM vertices from the learned texture coefficients
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        
        # Update the rendering and plot
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)

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
        
        plt.figure()
        plt.imshow(rendering)
        # break
        
        """
        Optimization simultaneously over the texture and lighting parameters
        """
        texParam2 = texParam.copy()
        
#        check_grad(opt.textureLightingCost, opt.textureLightingGrad, texParam, img, vertexCoords, B, m, renderObj)
        
        # Jointly optimize the texture and spherical harmonic lighting coefficients
        cost = np.zeros(10)
        for i in range(10):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, vertexCoords, B, m, renderObj, (1, 1), randomFaces), loss = 'soft_l1', max_nfev = 100)
            texParam2 = initTexLight['x']
            cost[i] = initTexLight.cost
            
        texCoef = texParam2[:m.numTex]
        lightCoef = texParam2[m.numTex:].reshape(9, 3)
        
        texture = generateTexture(vertexCoords, texParam2, m)
        
        # Render the 3DMM
        renderObj.updateVertexBuffer(np.r_[vertexCoords.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        plt.show()


        '''
        Optimization over shape, texture, and lighting
        '''
        # Need to do
        # break