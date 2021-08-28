#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from face2face.utils.transform import PCA
from face2face.utils.io import exportObj

import numpy as np
import re, os

def importObjFW(dirName, shape = 0, dataToImport = ['v', 'vt', 'f'], pose = 20):
    """
    Return the geometric and texture vertices along with the quadrilaterials containing the geometric and texture indices for all 150 testers of FaceWarehouse for a certain shape/pose/expression. Input (1) a string for the directory name that contains the folders 'Tester_1' through 'Tester_150', (2) an int for the shape number, which is in the range [0, 46] (1 neutral + 46 expressions), and (3) a list containing strings to indicate what part of the .obj file to read ('v' = geometric vertices, 'vt' = texture vertices, 'f' = face quadrilaterals).
    """
    # Number of observations (people/testers) in the dataset
    numTesters = 150
    
    # If input is just a single .obj file
    if dirName.endswith('.obj'):
        singleFile = True
        fName = dirName
    else:
        singleFile = False
        
        # Make sure directory name has final forward slash
        if not dirName.endswith('/'):
            dirName += '/'
    
    for i in range(numTesters):
        if (not singleFile) and pose == 47:
            fName = dirName + 'Tester_' + str(i+1) + '/Blendshape/shape_' + str(shape) + '.obj'
        elif (not singleFile) and pose == 20:
            fName = dirName + 'Tester_' + str(i+1) + '/TrainingPose/pose_' + str(shape) + '.obj'
        
        with open(fName) as fd:
            # Initialize lists to store the data from the .obj files
            v = []      # Geometric vertices (x, y, z)
            vt = []     # Texture vertices (U, V)
            f = []      # Face quadrilaterals
            for line in fd:
                if line.startswith('v ') and 'v' in dataToImport:
                    v.append([float(num) for num in line[2:].split(' ')])
                elif line.startswith('vt') and 'vt' in dataToImport and i == 0:
                    vt.append([float(num) for num in line[3:].split(' ')])
                elif line.startswith('f') and 'f' in dataToImport and i == 0:
                    f.append([int(ind) for ind in re.split('/| ', line[2:])])
                else:
                    continue
        
        if i == 0:
            geoV = np.empty((numTesters, len(v), 3))
            textV = np.empty((len(vt), 2))
            quad = np.empty((2, len(f), 4), dtype = 'int')
            
        # Store the data for each shape
        if 'vt' in dataToImport and i == 0:
            textV[:, :] = np.array(vt)
        if 'f' in dataToImport and i == 0:
            if len(f[0]) == 12:
                quad[0, :, :] = np.array(f)[:, [0, 3, 6, 9]]
                quad[1, :, :] = np.array(f)[:, [1, 4, 7, 10]]
            elif len(f[0]) == 3:
                quad = np.array(f)
            
        if 'v' in dataToImport and not singleFile:
            geoV[i, :, :] = np.array(v)
        elif 'v' in dataToImport and singleFile:
            geoV = np.array(v)
            break
        else:
            break
    
    # Select which data to return based on the dataToImport input
    objToData = {'v': geoV, 'vt': textV, 'f': quad}
    return [objToData.get(key) for key in dataToImport]

def generateModels(dirName, saveDirName = './'):
    """
    Generate eigenmodels of face meshes for (1) the neutral face and (2) the expressions. Save eigenvectors, eigenvalues, and means into .npy arrays.
    """
    if not dirName.endswith('/'):
        dirName += '/'
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    # Neutral face
    print('Loading neutral faces')
    vNeu = importObjFW(dirName, shape = 0, dataToImport = ['v'])[0]
    vNeu = np.reshape(vNeu, (150, vNeu.shape[1]*3))
    
    evalNeu, evecNeu, meanNeu = PCA(vNeu)
    
    
    np.save(saveDirName + 'idEval', evalNeu)
    np.save(saveDirName + 'idEvec', evecNeu)
    np.save(saveDirName + 'idMean', meanNeu)
    
    # Expressions (from the 46 expression blendshapes)
    vExp = np.empty((150*46, vNeu.shape[1]))
    for s in range(46):
        print('Loading expression %d' % (s+1))
        temp = importObjFW(dirName, shape = s+1, dataToImport = ['v'], pose = 47)[0]
        # Subtract the neutral shape from the expression shape for each test subject
        vExp[s*150: (s+1)*150, :] = np.reshape(temp, (150, vNeu.shape[1])) - vNeu
    
    
    evalExp, evecExp = PCA(vExp, numPC = 76)[:2]
    
    
    np.save(saveDirName + 'expEval', evalExp)
    np.save(saveDirName + 'expEvec', evecExp)

def saveLandmarks(dirName, saveDirName = './'):
    """
    Read the landmarks from the TrainingPoses and save them in a .npy file.
    """
    if not dirName.endswith('/'):
        dirName += '/'
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    landmarks = np.empty((20, 150, 74, 2))
    for pose in range(20):
        for tester in range(150):
            fName = 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.land'
            with open(dirName + fName, 'r') as fd:
                next(fd)
                landmarks[pose, tester, :, :] = np.array([[float(num) for num in line.split(' ')] for line in fd])
    
    np.save(saveDirName + 'landmarksTrainingPoses', landmarks)

def saveDepthMaps(dirName, saveDirName = './'):
    """
    Read the depth information from the Kinect .poses files and save them in a .npy file.
    """
    if not dirName.endswith('/'):
        dirName += '/'
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    depth = np.empty((20, 150, 480, 640), dtype = 'uint16')
    for pose in range(20):
        for tester in range(150):
            fName = 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.poses'
            with open(dirName + fName, 'rb') as fd:
                # First 4 bytes contain the frame number, and the next 640*480*3 contain the RGB data, so skip these
                fd.seek(4 + 1*640*480*3)
                
                # Each depth map value is 2 bytes (short)
                d = fd.read(2*640*480)
            
            # Convert bytes to int
            depth[pose, tester, :, :] = np.array([int.from_bytes(bytes([x, y]), byteorder = 'little') for x, y in zip(d[0::2], d[1::2])]).reshape((480, 640))
            
    np.save(saveDirName + 'depthMaps', depth)

def saveMasks(dirName, saveDirName = './masks/', mask = 'faceMask.obj', poses = 20):
    """
    Loop through the original 3D head models in the directory defined by dirName and extract the facial area defined by mask .obj file, saving the facial 3D models of the original 3D heads into new .obj files in the directory defined by saveDirName.
    """
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    # Loop through the poses/shapes
    for shape in range(poses):
        # The reference mask defining the facial region is based off of the first tester in pose/shape 0
        if shape == 0:
            v = importObjFW(dirName, shape, dataToImport = ['v'], pose = poses)[0]
            faceMask = importObjFW(mask, shape = 0)[0]
            idx = np.zeros(faceMask.shape[0], dtype = int)
            for i, vertex in enumerate(faceMask):
                idx[i] = np.where(np.equal(vertex, v[0, :, :]).all(axis = 1))[0]
        else:
            v = importObjFW(dirName, shape, dataToImport = ['v'], pose = poses)[0]
        
        v = v[:, idx, :]
        
        for tester in range(150):
            print('Processing shape %d for tester %d' % (shape+1, tester+1))
            if poses == 47:
                if not os.path.exists(saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/'):
                    os.makedirs(saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/')
                fName = saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/shape_' + str(shape) + '.obj'
            
            if poses == 20:
                if not os.path.exists(saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/'):
                    os.makedirs(saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/')
                fName = saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(shape) + '.obj'
            
            exportObj(v[tester, :, :], fNameIn = mask, fNameOut = fName)
