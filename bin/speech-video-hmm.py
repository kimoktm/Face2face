#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.utils.mesh import generateFace
from mm.utils.transform import rotMat2angle
from mm.utils.io import importObj, speechProc
from mm.models import MeshModel

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
from mayavi import mlab
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
from skimage import io
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, issparse
import tvtk

def mlab_imshowColor(im, alpha = 255, **kwargs):
    """
    Plot a color image with mayavi.mlab.imshow.
    im is a ndarray with dim (n, m, 3) and scale (0->255]
    alpha is a single number or a ndarray with dim (n*m) and scale (0->255]
    **kwargs is passed onto mayavi.mlab.imshow(..., **kwargs)
    """
    # Homogenous coordinate conversion
    im = np.concatenate((im, alpha * np.ones((im.shape[0], im.shape[1], 1), dtype = np.uint8)), axis = -1)
    colors = tvtk.UnsignedCharArray()
    colors.from_array(im.reshape(-1, 4))
    m_image = mlab.imshow(np.ones(im.shape[:2][::-1]))
    m_image.actor.input.point_data.scalars = colors
    
    mlab.draw()
    mlab.show()

    return

def animate(v, f, saveDir, t = None, alpha = 1):
    
    # Create the save directory for the images if it doesn't exist
    if not saveDir.endswith('/'):
        saveDir += '/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    # Render the mesh
    if t is None:
        tmesh = mlab.triangular_mesh(v[0, 0, :], v[0, 1, :], v[0, 2, :], f-1, scalars = np.arange(v.shape[2]), color = (1, 1, 1))
    
    # Add texture if given
    else:
        tmesh = mlab.triangular_mesh(v[0, 0, :], v[0, 1, :], v[0, 2, :], f-1, scalars = np.arange(v.shape[2]))
        if t.shape[1] is not 3:
            t = t.T
        tmesh.module_manager.scalar_lut_manager.lut.table = np.c_[(t * 255), alpha * 255 * np.ones(v.shape[2])].astype(np.uint8)
#        tmesh.actor.pro2perty.lighting = False
        
    # Change viewport to x-y plane and enforce orthographic projection
    mlab.view(0, 0, 'auto', 'auto')
    
    mlab.gcf().scene.parallel_projection = True
    
    # Save the first frame, then loop through the rest and save them
    mlab.savefig(saveDir + '00001.png', figure = mlab.gcf())
    tms = tmesh.mlab_source
    for i in range(1, v.shape[0]):
        fName = '{:0>5}'.format(i + 1)
        tms.set(x = v[i, 0, :], y = v[i, 1, :], z = v[i, 2, :])
        mlab.savefig(saveDir + fName + '.png', figure = mlab.gcf())

if __name__ == "__main__":
    
    # Change to relevant data directory
    os.chdir('/home/leon/f2f-fitting/data/obama/')
    
    # Specify file name of shiro audio file, number of frames in shiro video, and shiro video FPS
    fNameSiro = 'siroNorm.wav'
    numFramesSiro = 2882 #3744 #2260
    fpsSiro = 24
    
    # Process audio features for the source (shiro) audio file
    siroAudioVec, timeVecVideo = speechProc(fNameSiro, numFramesSiro, fpsSiro, return_time_vec = True)
    
    # Process audio features for the target (kuro) audio file
    fNameKuro = 'kuroNorm.wav'
    kuroAudioVec = speechProc(fNameKuro, numFramesSiro, fpsSiro, kuro = True)
    numFramesSiroKuro = kuroAudioVec.shape[1]
    
    # Cluster mfccs. Use 40 clusters -- 39 clusterable phonemes in American English
    M = 40
    X = siroAudioVec.T
    gmmObs = GaussianMixture(n_components = M, covariance_type = 'diag')
    gmmObs.fit(X)
    mfcc_classes = gmmObs.means_
    
    obsLabels_siro = gmmObs.predict(siroAudioVec.T)
    obsLabels_kuro = gmmObs.predict(kuroAudioVec.T)
    
    # Cluster the mouth vertices from the 3DMMs fitted to the video
    mouthFace = importObj('mouth.obj', dataToImport = ['f'])[0]
    mouthIdx = np.load('../../models/bfmMouthIdx.npy')
#    mouthVertices = np.zeros((numFramesSiro, mouthIdx.size, 3))
#    for i in range(numFramesSiro):
#        fName = '{:0>5}'.format(i + 1)
#        mouthVertices[i, :] = importObj('shapes/' + fName + '.obj', dataToImport = ['v'])[0][mouthIdx, :].flatten()
    
    mouthVertices = np.load('mouthVertices.npy')
    
    N = 100
    kShapes = KMeans(n_clusters = N)
    kShapes.fit(mouthVertices)
    
    stateShapes = kShapes.cluster_centers_
    stateLabels = kShapes.labels_
    
#    stateShapes2 = stateShapes.view().reshape((N, 3, mouthIdx.size), order = 'F')
#    tmesh = mlab.triangular_mesh(stateShapes2[0, 0, :], stateShapes2[0, 1, :], stateShapes2[0, 2, :], mouthFace-1, color = (1, 1, 1))
#    mlab.view(0, 0, 'auto', 'auto')
#    mlab.gcf().scene.parallel_projection = True
#    mlab.savefig('mouthStateShapes/00001.png', figure = mlab.gcf())
#    tms = tmesh.mlab_source
#    for i in range(1, N):
#        fName = '{:0>5}'.format(i + 1)
#        tms.set(x = stateShapes2[i, 0, :], y = stateShapes2[i, 1, :], z = stateShapes2[i, 2, :])
#        mlab.savefig('mouthStateShapes/' + fName + '.png', figure = mlab.gcf())
        
    
    # Find and cluster the features of the video in model-space
    m = MeshModel('../../models/bfm2017.npz')
    
    param = np.load('paramRTS2Orig.npy')
#    
#    N = 150
#    X = param[:, 80: -7]
#    kShapes = KMeans(n_clusters = N)
#    kShapes.fit(X)
#    
#    stateShapes = kShapes.cluster_centers_
#    stateLabels = kShapes.labels_
#    
    # Calculate initial state probabilities for states
    states, stateCounts = np.unique(stateLabels, return_counts = True)
    pi = stateCounts / stateLabels.size
    
    # Calculate transition probabilities using known clusters
    transition, transitionCounts = np.unique(np.c_[stateLabels[:-1], stateLabels[1:]], return_counts = True, axis = 0)
    A = np.zeros((N, N))
    A[transition[:, 0], transition[:, 1]] = transitionCounts
    A /= A.sum(1)[:, np.newaxis]
    
    # Calculate emission probabilities using known clusters
    B = np.zeros((N, M))
    for state in range(N):
        obsClass, classCount = np.unique(obsLabels_siro[stateLabels == state], return_counts = True)
        B[state, obsClass] = classCount
    B /= B.sum(1)[:, np.newaxis]
    
    # HMM stuff
    model = hmm.MultinomialHMM(n_components = N)
    model.startprob_ = pi
    model.transmat_ = A
    model.emissionprob_ = B
    
    # Try to reproduce siro
    stateSeq_siro = model.predict(obsLabels_siro.reshape(-1, 1))
    
    # Kuro
    stateSeq_kuro = model.predict(obsLabels_kuro.reshape(-1, 1))
    
#    np.save('siroGroundTruth', stateLabels)
#    np.save('siroStateSequence', stateSeq_siro)
#    np.save('kuroStateSequence', stateSeq_kuro)
#    np.save('shapeStateParams', stateShapes)
    
#    # Render and save pics
#    if not os.path.exists('stateShapes'):
#        os.makedirs('stateShapes')
#    for shape in range(N):
#        fName = '{:0>5}'.format(shape + 1)
#        exportObj(generateFace(np.r_[param[-1, :80], stateShapes[stateSeq_siro[shape], :], np.zeros(6), 1], m), f = m.face, fNameOut = 'stateShapes/' + fName)
#    
    selectedFrames = np.zeros(stateSeq_kuro.size, dtype = int)
    scaler = StandardScaler()
    normalizedRTS = scaler.fit_transform(param[:, -7:])
    for i in range(stateSeq_kuro.size):
        # Find the video frames that match to the current shape state
        frames = np.argwhere(stateLabels == stateSeq_kuro[i]).squeeze()
        
        # From these frames, find the frame that is closest to the i-th video frame in terms of rotation, translation, and scale
        candidateFramesRTS = normalizedRTS[frames, :]
        currentFrameRTS = normalizedRTS[i, :]
        
        NN = NearestNeighbors(n_neighbors = 1, metric = 'l2')
        
        NN.fit(candidateFramesRTS)
        distance, ind = NN.kneighbors(currentFrameRTS.reshape(1, -1))
        
        selectedFrames[i] = frames[ind.squeeze()]
#    np.save('kuroSelectedFrames', selectedFrames)
    
    """
    2nd HMM for temporal consistency
    """
    
#    N = 150
#    stateLabels = np.load('siroGroundTruth.npy')
#    stateSeq_siro = np.load('siroStateSequence.npy')
#    stateSeq_kuro = np.load('kuroStateSequence.npy')

    frameDifferences = metrics.pairwise.euclidean_distances(mouthVertices)

#    plt.figure()
#    plt.imshow(frameDifferences)
#    plt.figure()
#    plt.hist(mouthVertices.flatten(), bins = 'auto')
#    plt.figure()
#    plt.hist(frameDifferences.flatten(), bins = 'auto')
#    
#    nextFrame = np.r_[np.diag(frameDifferences, k = 1), frameDifferences[-1, -2]]
    minFrame = np.min(frameDifferences + frameDifferences.max()*np.eye(numFramesSiro), axis = 1)
    thres = minFrame + 200
#    
#    # Calculate pairwise difference between the frames as transition probabilities
#    videoVec = np.empty((numFramesSiro, mpimg.imread('orig/00001.png').size//3))
#    for i in range(numFramesSiro):
#        fName = '{:0>5}'.format(i + 1)
#        videoVec[i, :] = io.imread('orig/' + fName + '.png', as_grey = True).flatten()
#        
#    frameDifferences = metrics.pairwise.euclidean_distances(videoVec)
#    frameDifferences1 = np.load('frameDistanceMat.npy')
#    
#    plt.figure()
#    plt.imshow(frameDifferences1)
#    plt.figure()
#    plt.hist(frameDifferences.flatten(), bins = 'auto')
#    thres = np.min(frameDifferences + frameDifferences.max()*np.eye(numFramesSiro), axis = 1) + 30
#    
    transitionableFrames = frameDifferences < thres[:, np.newaxis]
    np.fill_diagonal(transitionableFrames, False)
    rowInd, colInd = np.nonzero(transitionableFrames)
    numTransitionableFrames = transitionableFrames.sum(1)
#    
#    plt.figure()
#    plt.hist(numTransitionableFrames, bins = numTransitionableFrames.max() - 1)
#    
    A = csc_matrix((np.exp(-0.1*frameDifferences[transitionableFrames]), (rowInd, colInd)), shape = (numFramesSiro, numFramesSiro))
    
    A.data /= np.take(A.sum(1).A1, A.indices)
#    
#    plt.figure()
#    plt.hist(A.data, bins = 100)
#    
#    plt.figure()
#    plt.imshow(A.toarray())
#    
    # Find the frames that match to each clustered 3DMM state and set a uniform PDF for these frames as the emission probabilities
#    state2frame = [None] * N
    B = 0.01*np.ones((numFramesSiro, N))
#    for i in range(N):
#        state2frame[i] = np.nonzero(stateLabels == i)[0].tolist()
#        B[state2frame[i], i] = 1. / len(state2frame[i])
#    
    B[np.arange(numFramesSiro), stateLabels] = 1
    B /= B.sum(1)[:, np.newaxis]
#    
    # Use a uniform PDF over all frames as the initial distribution
    pi = np.ones(numFramesSiro) / numFramesSiro
    
    # Set up the HMM
    model = hmm.MultinomialHMM(n_components = numFramesSiro)
    model.startprob_ = pi
    model.transmat_ = A.toarray()
    model.emissionprob_ = B
    
    frames = model.predict(stateSeq_kuro.reshape(-1, 1))
#    np.save('kuroSelectedFrames2.npy', frames)
    
    # Animations
    s = stateShapes.reshape((N, 3, mouthIdx.size), order = 'F')
    v = mouthVertices.reshape((numFramesSiro, 3, mouthIdx.size), order = 'F')
    animate(v[:240], mouthFace, 'color/1_origSiroFrames', m.texMean[:, mouthIdx])
    animate(s[stateLabels[:240]], mouthFace, 'color/2_siroKmeansStateSeq', m.texMean[:, mouthIdx])
    animate(s[stateSeq_siro[:240]], mouthFace, 'color/3_siroHMMStateSeq', m.texMean[:, mouthIdx])
    animate(s[stateSeq_kuro[:240]], mouthFace, 'color/4_kuroHMMStateSeq', m.texMean[:, mouthIdx])
    animate(v[selectedFrames[:240]], mouthFace, 'color/5_closestFramesToKuroFromSiroViaRTS', m.texMean[:, mouthIdx])
    animate(v[frames[:240]], mouthFace, 'color/6_closestFramesToKuroFromSiroViaHMM', m.texMean[:, mouthIdx])