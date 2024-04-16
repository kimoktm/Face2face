#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import h5py
from sklearn.neighbors import NearestNeighbors

def processBFM2017(fName, fNameLandmarks):
    """
    Read the face models and landmarks from the Basel Face Model 2017 dataset. Input the filename of the .h5 file and the filename of a .txt file containing the text detailing the landmark locations.
    """
    data = h5py.File(fName, 'r')
    
    # Identity
    idMean = np.empty(data.get('/shape/model/mean').shape)
    data.get('/shape/model/mean').read_direct(idMean)
    idVar = np.empty(data.get('/shape/model/noiseVariance').shape)
    data.get('/shape/model/noiseVariance').read_direct(idVar)
    idEvec = np.empty(data.get('/shape/model/pcaBasis').shape)
    data.get('/shape/model/pcaBasis').read_direct(idEvec)
    idEval = np.empty(data.get('/shape/model/pcaVariance').shape)
    data.get('/shape/model/pcaVariance').read_direct(idEval)
    
    # Expression
    expMean = np.empty(data.get('/expression/model/mean').shape)
    data.get('/expression/model/mean').read_direct(expMean)
    expVar = np.empty(data.get('/expression/model/noiseVariance').shape)
    data.get('/expression/model/noiseVariance').read_direct(expVar)
    expEvec = np.empty(data.get('/expression/model/pcaBasis').shape)
    data.get('/expression/model/pcaBasis').read_direct(expEvec)
    expEval = np.empty(data.get('/expression/model/pcaVariance').shape)
    data.get('/expression/model/pcaVariance').read_direct(expEval)
    
    # Texture
    texMean = np.empty(data.get('/color/model/mean').shape)
    data.get('/color/model/mean').read_direct(texMean)
    texVar = np.empty(data.get('/color/model/noiseVariance').shape)
    data.get('/color/model/noiseVariance').read_direct(texVar)
    texEvec = np.empty(data.get('/color/model/pcaBasis').shape)
    data.get('/color/model/pcaBasis').read_direct(texEvec)
    texEval = np.empty(data.get('/color/model/pcaVariance').shape)
    data.get('/color/model/pcaVariance').read_direct(texEval)
    
    # Triangle face indices
    face = np.empty(data.get('/shape/representer/cells').shape, dtype = 'int')
    data.get('/shape/representer/cells').read_direct(face)
    
    # Find vertex indices corresponding to the 40 given landmark vertices
    points = np.empty(data.get('/shape/representer/points').shape)
    data.get('/shape/representer/points').read_direct(points)
    
    # with open(fNameLandmarks, 'r') as fd:
    #     landmark = []
    #     for line in fd:
    #         landmark.append([x for x in line.split(' ')])
    
    # landmark = np.array(landmark)
    # landmarkName = landmark[:, 0].tolist()
    # landmark = landmark[:, 2:].astype('float')
    
    # NN = NearestNeighbors(n_neighbors = 1, metric = 'l2')
    # NN.fit(points.T)
    # landmarkInd = NN.kneighbors(landmark)[1].squeeze()
    
    # Reshape to be compatible with fitting code
    numVertices = idMean.size // 3
    idMean = idMean.reshape((3, numVertices), order = 'F')
    idEvec = idEvec.reshape((3, numVertices, 199), order = 'F')
    expMean = expMean.reshape((3, numVertices), order = 'F')
    expEvec = expEvec.reshape((3, numVertices, 100), order = 'F')
    texMean = texMean.reshape((3, numVertices), order = 'F')
    texEvec = texEvec.reshape((3, numVertices, 199), order = 'F')
    
    # Find the face indices associated with each vertex (for norm calculation)
    vertex2face = np.array([np.where(np.isin(face, vertexInd).any(axis = 0))[0][0] for vertexInd in range(numVertices)])
    face = face.T
    
    # Save into an .npz uncompressed file
    # np.savez('./models/bfm2017', face = face, idMean = idMean, idEvec = idEvec, idEval = idEval, expMean = expMean, expEvec = expEvec, expEval = expEval, texMean = texMean, texEvec = texEvec, texEval = texEval, landmark = landmark, landmarkInd = landmarkInd, landmarkName = landmarkName, numVertices = numVertices, vertex2face = vertex2face)
    np.savez('./models/bfm2017', face = face, idMean = idMean, idEvec = idEvec, idEval = idEval, expMean = expMean, expEvec = expEvec, expEval = expEval, texMean = texMean, texEvec = texEvec, texEval = texEval, numVertices = numVertices, vertex2face = vertex2face)


if __name__ == '__main__':

    processBFM2017('./models/model2017-1_face12_nomouth.h5', '')
