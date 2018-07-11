#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions that concern operations on 3DMMs. Perhaps these will be integrated into the MeshModel class later on.
"""
import numpy as np
from .transform import rotMat2angle, sh9
from sklearn.preprocessing import normalize


def generateFace(param, model, ind = None):
    """Generates vertex coordinates based on the 3DMM eigenmodel and the shape identity parameters, the shape facial expression parameters, and the similarity transform parameters.
    
    Args:
        param (ndarray): Contains the concatenation of the shape identity parameters, the shape facial expression parameters, the three Euler angles, the three translation vector terms in Cartesian space, and a scaling factor. The amount of shape parameters should match the number of shape eigenvectors in the 3DMM.
        model (MeshModel): 3DMM MeshModel class object
        ind (ndarray): Optional, a list of certain vertex indices in the 3DMM to return
    
    Returns:
        ndarray: vertex coordinates
    """
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]
    
    # Rotation Euler angles, translation vector, scaling factor
    R = rotMat2angle(param[model.numId + model.numExp:][:3])
    t = param[model.numId + model.numExp:][3: 6]
    s = param[model.numId + model.numExp:][6]

    # The eigenmodel, before rigid transformation and scaling
    if ind is None:
        model = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)
    else:
        model = model.idMean[:, ind] + np.tensordot(model.idEvec[:, ind, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, ind, :], expCoef, axes = 1)

    # After rigid transformation and scaling
    return s * np.dot(R, model) + t[:, np.newaxis]

def generateTexture(vertexCoord, texParam, model):
    """Generates vertex colors based on the 3DMM eigenmodel, the vertex coordinates, and the texture parameters and spherical harmonic lighting parameters.
    
    Args:
        vertexCoord (ndarray): Vertex coordinates for the 3DMM, (3, numVertices)
        texParam (ndarray): Contains the concatenation of the texture parameters and a flattened version of the spherical harmonic lighting parameters such as the lighting parameters for each color channel are grouped together. The amount of texture parameters should match the number of texture eigenvectors in the 3DMM.
        model (MeshModel): 3DMM MeshModel class object
    
    Returns:
        ndarray, (3, numVertices): vertex RGB colors
    """
    texCoef = texParam[:model.texEval.size]
    shCoef = texParam[model.texEval.size:].reshape(9, 3)
    vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)

    # Evaluate spherical harmonics at face shape normals
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])

    I = np.empty((3, model.numVertices))
    for c in range(3):
        I[c, :] = np.dot(shCoef[:, c], sh) * vertexColor[c, :]

    return I

def barycentricReconstruction(vertices, pixelFaces, pixelBarycentricCoords, indexData):
    """Reconstructs per-pixel attributes from a barycentric combination of the vertices in the triangular face underlying the pixel.
    
    Args:
        vertices (ndarray): An array of a certain per-vertex attribute, e.g., vertex coordinates, vertex colors, spherical harmonic bases, etc., (n, numVertices)
        pixelFaces (ndarray): The triangular face IDs for each pixel where the 3DMM is drawn, (numPixels,)
        pixelBarycentricCoords (ndarray): The barycentric coordinates of the vertices on the triangular face underlying each pixel where the 3DMM is drawn, (numPixels, 3)
        indexData (ndarray): An array containing the vertex indices for each face, (numFaces, 3)

    Returns:
        ndarray: The per-pixel barycentric reconstruction of the desired per-vertex attribute
    """
    pixelVertices = indexData[pixelFaces, :]
    
    if len(vertices.shape) is 1:
        vertices = vertices[np.newaxis, :]

    numChannels = vertices.shape[0]

    colorMat = vertices[:, pixelVertices.flat].reshape((numChannels, 3, pixelFaces.size), order = 'F')

    return np.einsum('ij,kji->ik', pixelBarycentricCoords, colorMat)

def calcNormals(vertexCoord, model):
    """Calculates the per-vertex normal vectors for a model given shape coefficients.
    
    Args:
        vertexCoord (ndarray): Vertex coordinates for the 3DMM, (3, numVertices)
        model (MeshModel): 3DMM MeshModel class object
    
    Returns:
        ndarray: Per-vertex normal vectors
    """
    faceNorm = np.cross(vertexCoord[:, model.face[:, 0]] - vertexCoord[:, model.face[:, 1]], vertexCoord[:, model.face[:, 0]] - vertexCoord[:, model.face[:, 2]], axisa = 0, axisb = 0)

    vNorm = np.array([np.sum(faceNorm[faces, :], axis = 0) for faces in model.vertex2face])

    return normalize(vNorm, norm = 'l2')


# TO DO: only use pixel faces
def calcFaceNormals(vertexCoord, model, pixelFaces = None):
    """Calculates the per-vertex normal vectors for a model given shape coefficients.
    
    Args:
        vertexCoord (ndarray): Vertex coordinates for the 3DMM, (3, numVertices)
        model (MeshModel): 3DMM MeshModel class object
    
    Returns:
        ndarray: Per-vertex normal vectors
    """
    faceNorm = np.cross(vertexCoord[:, model.face[:, 0]] - vertexCoord[:, model.face[:, 1]], vertexCoord[:, model.face[:, 0]] - vertexCoord[:, model.face[:, 2]], axisa = 0, axisb = 0)

    if pixelFaces is not None:
        faceNorm = faceNorm[np.unique(pixelFaces)]

    return normalize(faceNorm, norm = 'l2')


def subdivide(v, f):
    """Uses Catmull-Clark subdivision to subdivide a 3DMM with quadrilateral faces, increasing the number of faces by 4 times.
    
    Args:
        v (ndarray): Vertex coordinates for the 3DMM, (4, numVertices)
        f (ndarray): An array containing the vertex indices for each quadrilateral face, (numFaces, 4)
    
    Returns:
        tuple: Subdivided vertex coordinates and array of vertex indices
    """
    from collections import defaultdict
    from itertools import chain, compress

    # Make v 3D if it isn't, for my convenience
    if len(v.shape) != 3:
        v = v[np.newaxis, :, :]
    
    # Check to make sure f is 2D (only shape info) and indices start at 0
    if len(f.shape) != 2:
        f = f[0, :, :]
    if np.min(f) != 0:
        f = f - 1
        
    # Find the edges in the input face mesh
    edges = np.c_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 3]], f[:, [3, 0]]]
    edges = np.reshape(edges, (4*f.shape[0], 2))
    edges = np.sort(edges, axis = 1)
    edges, edgeInd = np.unique(edges, return_inverse = True, axis = 0)
    edges = [frozenset(edge) for edge in edges]
    
    # Map from face index to sets of edges connected to the face
    face2edge = [[frozenset(edge) for edge in np.c_[face[:2], face[1:3], face[2:4], face[[-1, 0]]].T] for face in f]
    
    # Map from sets of edges to face indices
    edge2face = defaultdict(list)
    for faceInd, edgesOnFace in enumerate(face2edge):
        for edge in edgesOnFace:
            edge2face[edge].append(faceInd)
    
    # Map from vertices to the faces they're connected to
    vertex2face = [np.where(np.isin(f, vertexInd).any(axis = 1))[0].tolist() for vertexInd in range(v.shape[1])]
    
    # Map from vertices to the edges they're connected to
    vertex2edge = [list(compress(edges, [vertexInd in edge for edge in edges])) for vertexInd in range(v.shape[1])]
    
    # Number of faces connected to each vertex (i.e. valence)
    nFaces = np.array([np.isin(f, vertexInd).any(axis = 1).sum() for vertexInd in range(v.shape[1])])
    
    # Number of edges connected to each vertex
    nEdges = np.array([len(vertex2edge[vertexInd]) for vertexInd in range(v.shape[1])])
    
    # Loop thru the vertices of each tester's face to find the new set of vertices
    for tester in range(v.shape[0]):
        print('Calculating new vertices for tester %d' % (tester + 1))
        # Face points: the mean of the vertices on a face
        facePt = np.array([np.mean(v[tester, vertexInd, :], axis = 0) for vertexInd in f])
        
        # Edge points
        edgePt = np.empty((len(edges), 3))
        for i, edge in enumerate(edges):
            # If an edge is only associated with one face, then it is on a border of the 3D model. The edge point is thus the midpoint of the vertices defining the edge.
            if len(edge2face[edge]) == 1:
                edgePt[i, :] = np.mean(v[tester, list(edge), :], axis = 0)
            
            # Else, the edge point is the mean of (1) the face points of the two faces adjacent to the edge and (2) the midpoint of the vertices defining the edge.
            else:
                edgePt[i, :] = np.mean(np.r_[facePt[edge2face[edge], :], v[tester, list(edge), :]], axis = 0)
        
        # New coordinates: loop thru each vertex P of the original vertices to calc
        newPt = np.empty(v.shape[1: ])
        for i, P in enumerate(v[tester, :, :]):
            # If P is not on the border
            if nFaces[i] == nEdges[i]:
                # Mean of the face points from the faces surrounding P
                F = np.mean(facePt[vertex2face[i], :], axis = 0)
                
                # Mean of the edge midpoints from the edges connected to P
                R = np.mean(v[tester, list(chain.from_iterable(vertex2edge[i])), :], axis = 0)
                
                # The new coordinates of P is a combination of F, R, and P
                newPt[i, :] = (F + 2*R + (nFaces[i] - 3)*P)/nFaces[i]
                
            # Otherwise, P is on the border
            else:
                # For the edges connected to P, find the edges on the border
                borderEdge = [len(edge2face[edge]) == 1 for edge in vertex2edge[i]]
                
                # The midpoints of these edges on the border
                R = v[tester, list(chain.from_iterable(compress(vertex2edge[i], borderEdge))), :]
                
                # The new coordinates of P is the mean of R and P
                newPt[i, :] = np.mean(np.r_[R, P[np.newaxis, :]], axis = 0)
        
        # Save the result
        if tester == 0:
            vNew = np.empty((v.shape[0], facePt.shape[0] + edgePt.shape[0] + newPt.shape[0], 3))
            
        vNew[tester, :, :] = np.r_[facePt, edgePt, newPt]
    
    # Form the new faces
    fNew = np.c_[f.flatten() + facePt.shape[0] + edgePt.shape[0], edgeInd + facePt.shape[0], np.repeat(np.arange(facePt.shape[0]), 4), edgeInd.reshape((edgeInd.shape[0]//4, 4))[:, [3, 0, 1, 2]].flatten() + facePt.shape[0]] + 1
    
    return vNew, fNew


def writePly(file_name, vertices, faces, colors, landmarks = None):
    print('Saving ' + file_name)

    num_vertices = vertices.shape[1]
    num_faces = faces.shape[0]

    file = open(file_name, 'w')
    file.write("ply\nformat ascii 1.0\n")
    file.write("element vertex %d\n" % num_vertices)
    file.write("property float x\nproperty float y\nproperty float z\n")
    if landmarks is not None:
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")

    file.write("element face %d\n" % num_faces)
    file.write("property list uchar int vertex_index\nend_header\n")

    for i in range(num_vertices):
        x, y, z = vertices[:, i]
        file.write(str(x) + ' ' + str(y) + ' ' + str(z))
        if landmarks is not None:
            if i in landmarks:
                file.write(' 0 0 0')
            else:
                r, g, b = colors[:, i] * 255
                file.write(' ' + str(int(r)) + ' ' + str(int(g)) + ' ' + str(int(b)))
        file.write('\n')

    for i in range(num_faces):
        v1, v2, v3 = faces[i, :]
        file.write('3 ' + str(v1) + ' ' + str(v2) + ' ' + str(v3)+'\n')

    file.close()


# def writeOBJ(file_name, vertices, normals, FV, FN):   
#     print('Saving ' + file_name)
    
#     vertexToken = 'v'
#     normalToken = 'vn'
#     textureToken = 'vt'
#     faceToken = 'f'

#     V = vertices.reshape((-1,3))
#     N = normals.reshape((-1,3))
    
#     num_vertices = V.shape[0]
#     num_normals = N.shape[0]
#     num_faces = FV.shape[0]
    
#     file = open(file_name, 'w')
    
#     file.write("o Mesh\n")
    
#     for i in range(num_vertices):
#         x, y, z = V[i, :]
#         file.write(vertexToken + ' ' + str(x) + ' ' + str(y) + ' ' + str(z)+'\n')
    
#     for i in range(num_normals):
#         x, y, z = N[i, :]
#         file.write(normalToken + ' ' + str(x) + ' ' + str(y) + ' ' + str(z)+'\n')

#     for i in range(num_faces):
#         v1, v2, v3 = FV[i, :]
#         n1, n2, n3 = FN[i, :]
        
#         f1 = str(v1) + '/' + str(n1)
#         f2 = str(v2) + '/' + str(n2)
#         f3 = str(v3) + '/' + str(t3)
        
#         file.write(faceToken + ' ' + f1 + ' ' + f2 + ' ' + f3 + '\n')

#     file.close()

