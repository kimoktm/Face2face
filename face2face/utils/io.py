#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions that concern the input and output of files.
"""
import numpy as np
import re

def importObj(fName, dataToImport = ['v', 'f']):
    """Returns the shape vertices and the list of vertex indices for each mesh face.
    
    Args:
        fName (str): Filename of .obj file
        dataToImport (list): A list containing strings that indicate what part of the .obj file to read (``v`` = geometric vertices, ``f`` = face indices). Note that all of the .obj files for a given 3DMM have the same face indices.
    
    Returns:
        ndarray or tuple: the vertex coordinates, the vertex indices for each face, or both
    """
        
    with open(fName) as fd:
        # Initialize lists to store the data from the .obj files
        v = []      # Geometric vertices (x, y, z)
        f = []      # Faces
        for line in fd:
            if line.startswith('v ') and 'v' in dataToImport:
                v.append([float(num) for num in line[2:].split(' ')])
            elif line.startswith('f') and 'f' in dataToImport:
                f.append([int(ind) for ind in re.split('/| ', line[2:])])
            else:
                continue
        
    # Store the data for each shape
    if 'f' in dataToImport:
        f = np.array(f)
        
        if 'v' in dataToImport:
            v = np.array(v)
            return v, f
        else:
            return f
    
    elif 'v' in dataToImport:
        v = np.array(v)
        return v
    
def exportObj(v, c = None, vt = None, f = None, fNameIn = None, fNameOut = 'test.obj'):
    """Writes vertices to an .obj file.
    
    Args:
        v (ndarray): Vertex coordinates
        c (ndarray): Optional, vertex colors
        vt (ndarray): Optional, mapping of vertex coordinates to UV coordinates
        f (ndarray): Optional, array of vertex indices that correspond to each mesh face
        fNameIn (str): Optional, filename of .obj file to overwrite certain attributes with
        fNameOut (str): Optional, output filename
    
    Returns:
        .obj file
    """
    # Make sure x, y, z vertex coordinates are along the columns
    if v.shape[1] != 3:
        v = v.T
    
    # Add the .obj extension if necessary
    if not fNameOut.endswith('.obj'):
        fNameOut += '.obj'
    
    # If user provides an .obj file for the face specification
    if fNameIn is not None:
        if not fNameIn.endswith('.obj'):
            fNameIn += '.obj'
            
        with open(fNameIn, 'r') as fi:
            with open(fNameOut, 'w') as fo:
                # Initialize counter for the vertex index
                vertexInd = 0
                
                # Iterate through the lines in the template file
                for line in fi:
                    # If a line starts with 'v ', then we replace it with the vertex coordinate with the index corresponding to the line number
                    if line.startswith('v '):
                        # Keep 6 digits after the decimal to follow the formatting in the template file
                        fo.write('v ' + '{:.6f}'.format(v[vertexInd, 0]) + ' ' + '{:.6f}'.format(v[vertexInd, 1]) + ' ' + '{:.6f}'.format(v[vertexInd, 2]) + '\n')
                        
                        # Increment the vertex index counter
                        vertexInd += 1
                    
                    elif line.startswith('vn'):
                        continue
                    
                    # Else, the line must end with 'vt' or 'f', so we copy from the template file because those coordinates are in correspondence
                    elif line.startswith('vt'):
                        fo.write(line)
                    
                    elif line.startswith('f'):
                        f = [int(ind) for ind in re.split('/| ', line[2:])]
                        fo.write('f ' + str(f[0]) + '/' + str(f[1]) + '/0 ' + str(f[3]) + '/' + str(f[4]) + '/0 ' + str(f[6]) + '/' + str(f[7]) + '/0 ' + str(f[9]) + '/' + str(f[10]) + '/0\n')
                    
                    else:
                        continue
    # If user provides a NumPy array for the face specification, or doesn't provide one at all (just to write out the vertices)
    else:
        with open(fNameOut, 'w') as fo:
            if c is None:
                for vertex in v:
                    fo.write('v ' + '{:.6f}'.format(vertex[0]) + ' ' + '{:.6f}'.format(vertex[1]) + ' ' + '{:.6f}'.format(vertex[2]) + '\n')
            else:
                if c.shape[1] != 3:
                    c = c.T
                for i in range(v.shape[0]):
                    fo.write('v ' + '{:.6f}'.format(v[i, 0]) + ' ' + '{:.6f}'.format(v[i, 1]) + ' ' + '{:.6f}'.format(v[i, 2]) + ' ' + '{:.6f}'.format(c[i, 0]) + ' ' + '{:.6f}'.format(c[i, 1]) + ' ' + '{:.6f}'.format(c[i, 2]) + '\n')
            
            if vt is not None:
                for text in vt:
                    fo.write('vt ' + '{:.6f}'.format(text[0]) + ' ' + '{:.6f}'.format(text[1]) + '\n')
            
            if f is not None:
                # obj face indexing starts at 1
                if np.min(f) == 0:
                    f = f + 1
                if f.shape[1] == 4:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n')
                elif f.shape[1] == 3:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
