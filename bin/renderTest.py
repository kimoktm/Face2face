#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.utils.opengl import Render
from mm.utils.mesh import generateFace
from mm.models import MeshModel
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

if __name__ == '__main__':
    
    # Load the first image from the video
    frame = 0
    img = io.imread('../data/obama/orig/%05d.png' % (frame + 1))
    img = img_as_float(img)
    width = img.shape[1]
    height = img.shape[0]
    
    # Load the 3DMM parameters that fit the 3DMM to each video frame
    param = np.load('../data/obama/paramRTS2Orig.npy')
    
    # Load the mesh model
    m = MeshModel('../models/bfm2017.npz')
    
    # Generate the vertex coordinates from the mesh model and the parameters
    vertexCoords = generateFace(param[frame, :], m).T
    
    # Use the mean vertex colors just for illustrative purposes
    vertexColors = m.texMean.T
    
    # Concatenate the vertex coordinates and colors-- this is how they will be inputted into the Render object
    meshData = np.r_[vertexCoords, vertexColors]
    
    # Initialize an OpenGL Render object and render the 3DMM with the corresponding video frame in the background
    r = Render(width, height, meshData, m.face, indexed = False, img = img)
    r.render()
    
    # Grab the rendering from the video card
    rendering = r.grabRendering()
    
    # You can also get other parameters from the video card, such as the pixels where the 3DMM is rendered on, the index of the triangular face that contributes to the color of each pixel of these pixels, and the barycentric coordinates of such a triangular face such that the barycentric combination of the three vertex attributes (e.g. color) for this triangular face forms the color in the rendered pixel
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = r.grabRendering(return_info = True)
    
    # Plot the rendering
    plt.figure()
    plt.imshow(rendering)
    
    # Loop through some frames in the video to render some more 3DMMs
    for frame in range(1, 52, 10):
        img = io.imread('../data/obama/orig/%05d.png' % (frame + 1))
        img = img_as_float(img)
        
        vertexCoords = generateFace(param[frame, :], m).T
        meshData = np.r_[vertexCoords, vertexColors]
        
        # Update the video card with the new mesh data for the current frame
        r.updateVertexBuffer(meshData)
        
        # Erase the current rendering to prepare for the new rendering
        r.resetFramebufferObject()
        
        # And then render and plot the rendering
        r.render()
        rendering = r.grabRendering()
        plt.figure()
        plt.imshow(rendering)