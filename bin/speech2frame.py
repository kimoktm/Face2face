#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.utils.mesh import generateFace
from mm.utils.transform import rotMat2angle
from mm.utils.io import importObj, speechProc
from mm.models import MeshModel
from mm.utils.visualize import animate

import glob, os, json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import networkx as nx
        
if __name__ == "__main__":
    
    # Change to relevant data directory
    os.chdir('/home/leon/f2f-fitting/data/obama/')
    
    # Specify file name of shiro audio file, number of frames in shiro video, and shiro video FPS
    fNameSiro = 'siroNorm.wav'
    numFramesSiro = 2882 #3744 #2260
    fpsSiro = 24
    
    # Process audio features for the source (shiro) audio file
    siroAudioVec, timeVecVideo = speechProc(fNameSiro, numFramesSiro, fpsSiro, return_time_vec = True)
    
    # Create a kNN fitter to find the k closest siro audio features
    k = 20
    NN = NearestNeighbors(n_neighbors = k, metric = 'l2')
    NN.fit(siroAudioVec.T)
    
    """
    Initialize 3DMM, relevant OpenPose landmark indices, etc.
    """
    # Load 3DMM
    m = MeshModel('../../models/bfm2017.npz')
    
    # Load 3DMM parameters for the shiro video, scaling some for a distance measure
    scaler = StandardScaler()
    param = np.load('paramRTS2Orig.npy')    # Parameters to orthographically project 3DMM onto shiro frame images
    expCoef = scaler.fit_transform(param[:, m.numId: m.numId + m.numExp])
    angles = param[:, m.numId + m.numExp: m.numId + m.numExp + 3]
    trans = scaler.fit_transform(param[:, m.numId + m.numExp + 3: m.numId + m.numExp + 5])
    R = np.empty((numFramesSiro, 3, 3))
    for i in range(numFramesSiro):
        R[i, ...] = rotMat2angle(angles[i, :])
    
    # Load OpenPose 2D landmarks for the siro video
    lm = np.empty((numFramesSiro, 70, 2))
    for i in range(numFramesSiro):
        with open('landmark/' + '{:0>5}'.format(i+1) + '.json', 'r') as fd:
            lm[i, ...] = np.array([l[0] for l in json.load(fd)], dtype = int).squeeze()[:, :2]
    
    # These pairs of OpenPose landmark indices correspond to certain features that we want to measure, such as the distance between the lower and upper lips, eyelids, etc.
    targetLMPairs = np.array([[42, 47], [43, 46], [44, 45], [30, 36], [42, 45], [44, 47], [25, 29], [26, 28], [19, 23], [20, 22]])
            
    # Get corresponding landmark pairs on the 3DMM
    sourceLMPairs = m.sourceLMInd[targetLMPairs]
    
    # Get the unique landmarks in these landmark pairs
    uniqueSourceLM, uniqueInv = np.unique(sourceLMPairs, return_inverse = True)
    
    # Load mouth region from 3DMM for animation
    mouthIdx = np.load('../../models/bfmMouthIdx.npy')
    mouthVertices = np.load('mouthVertices.npy')
    mouthFace = importObj('mouth.obj', dataToImport = ['f'])
    
    """
    Loop through the kuro (target) audio files of interest and find the shortest path sequence of shiro video frames to reenact the target audio file
    """
    
    # Loop through each target audio file
    for fNameKuro in glob.glob('condition_enhanced/cleaned/*.wav'):
        fNameKuro = 'condition_enhanced/cleaned/7_EJF101_ESPBOBAMA1_00101_V01_T01.wav'
        kuroAudioVec = speechProc(fNameKuro, numFramesSiro, fpsSiro, kuro = True)
        numFramesKuro = kuroAudioVec.shape[1]
        
        distance, ind = NN.kneighbors(kuroAudioVec.T)
        
        # Enforce similarity in similarity transform parameters from candidate frames to original video frames
        Dp = np.empty((numFramesKuro, k))
        for q in range(numFramesKuro):
            c = ind[q, :]
            Dp[q, :] = np.linalg.norm(trans[q, :] - trans[c, :], axis = 1) + np.linalg.norm(R[q, ...] - R[c, ...], axis = (1, 2))
            
        # Transition between candidate frames should have similar 3DMM landmarks and expression parameters
        mmLm = np.empty((numFramesSiro, 3, uniqueSourceLM.size))
        for t in range(numFramesSiro):
            mmLm[t] = generateFace(param[t, :], m, ind = uniqueSourceLM)
        mmLm = mmLm[..., uniqueInv[::2]] - mmLm[..., uniqueInv[1::2]]
        mmLmNorm = np.linalg.norm(mmLm, axis = 1)
        
        Dm = np.empty((numFramesKuro - 1, k, k))
        weights = np.empty((numFramesKuro - 1, k, k))
        for t in range(numFramesKuro - 1):
            for c1 in range(k):
                Dm[t, c1] = np.linalg.norm(mmLmNorm[ind[t, c1], :] - mmLmNorm[ind[t+1, :], :], axis = 1) + np.linalg.norm(expCoef[ind[t, c1]] - expCoef[ind[t+1, :], :], axis = 1)
                
    #            np.exp(-np.fabs(timeVecVideo[ind[t, c1]] - timeVecVideo[ind[t+1, :]])**2)
                
                weights[t, c1] = Dm[t, c1] + Dp[t, c1] + Dp[t+1, :] + distance[t, c1] + distance[t+1, :]
        
        # Create DAG and assign edge weights from distance matrix
        G = nx.DiGraph()
        for i in range(numFramesKuro - 1):
            left = np.arange(i*k, (i+1)*k)
            right = np.arange((i+1)*k, (i+2)*k)
            G.add_nodes_from(left)
            G.add_nodes_from(right)
            G.add_weighted_edges_from((u, v, weights[i, u - i*k, v - (i+1)*k]) for u in left for v in right)
        
        # Use A* shortest path algorithm to find the distances from each of the k source nodes to each of the k terminal nodes
        astarLength = np.empty((k, k))
        for s in range(k):
            for t in range(k):
                astarLength[s, t] = nx.astar_path_length(G, s, right[t])
        
        # Find the optimal path with the minimum distance of the k^2 paths calculated above
        s, t = np.unravel_index(astarLength.argmin(), (k, k))
        optPath = nx.astar_path(G, s, right[t])
        optPath = np.unravel_index(optPath, (numFramesKuro, k))
        optPath = ind[optPath[0], optPath[1]]
        
        # Save the optimal path of shiro video frame indices as an .npy file
#        if not os.path.exists('graphOptPath'):
#            os.makedirs('graphOptPath')
#        np.save('graphOptPath/' + os.path.splitext(os.path.basename(fNameKuro))[0], optPath)
        
        # Animate the reenactment and save
        v = mouthVertices.reshape((numFramesSiro, 3, mouthIdx.size), order = 'F')
        animate(v[optPath], mouthFace, 'temp/' + os.path.splitext(os.path.basename(fNameKuro))[0], m.texMean[:, mouthIdx])
        break