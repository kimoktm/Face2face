#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["QT_API"] = "pyqt"
import numpy as np
#from sklearn.neighbors import NearestNeighbors
#from sklearn.mixture import GaussianMixture
#from sklearn.cluster import KMeans
#from sklearn import metrics
from mayavi import mlab
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import skimage.io
#from pylab import savefig

def onpick3(event):
    """
    Interactive clicking for pyplot scatter plots.
    Example:
    fig, ax = plt.subplots()
    x = ...
    y = ...
    col = ax.scatter(x, y, s = 1, picker = True)
    fig.canvas.mpl_connect('pick_event', onpick3)
    """
    ind = event.ind
    print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))
    
def mlab_imshowColor(im, alpha = 255, **kwargs):
    """
    Plot a color image with mayavi.mlab.imshow.
    im is a ndarray with dim (n, m, 3) and scale (0->255]
    alpha is a single number or a ndarray with dim (n*m) and scale (0->255]
    **kwargs is passed onto mayavi.mlab.imshow(..., **kwargs)
    """
    from tvtk.api import tvtk
    
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
#        tmesh.actor.property.lighting = False
        
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
    
    mlab.close(all = True)
        
if __name__ == "__main__":
    
    m = Bunch(np.load('./models/bfm2017.npz'))
    m.idEvec = m.idEvec[:, :, :80]
    m.idEval = m.idEval[:80]
    m.expEvec = m.expEvec[:, :, :76]
    m.expEval = m.expEval[:76]
    m.texEvec = m.texEvec[:, :, :80]
    m.texEval = m.texEval[:80]
    
    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
    
    os.chdir('/home/leon/f2f-fitting/obama/')
    numFrames = 2882
    
#    selectedFrames = np.load('kuroSelectedFrames.npy')
#    plt.ioff()
#    for i in range(selectedFrames.size):
#        fName = '{:0>5}'.format(i+1)
#        fNameSelected = '{:0>5}'.format(selectedFrames[i]+1)
#    #    frameOrig = skimage.io.imread('orig/' + fName + '.png', as_grey = True)
#        frameOrig = mpimg.imread('orig/' + fName + '.png')
#        frameSelected = mpimg.imread('orig/' + fNameSelected + '.png')
#        
#        plt.figure()
#        plt.imshow(frameOrig, alpha = 1)
#        plt.imshow(frameSelected, alpha = 0.5)
#        plt.title('Frame: %d, Selected Frame: %d' % (i+1, selectedFrames[i] + 1))
#        if not os.path.exists('../selectedFrames'):
#            os.makedirs('../selectedFrames')
#        savefig('../selectedFrames/' + fName + '.png', bbox_inches='tight')
#        plt.close('all')
    
    # Landmarks
    #param = np.load('param.npy')
    #plt.ioff()
    #for i in range(numFrames):
    #    fName = '{:0>5}'.format(i + 1)
    #    imgScaled = mpimg.imread('scaled/' + fName + '.png')
    #    
    #    source = generateFace(param[i, :], m)
    #    plt.figure()
    #    plt.imshow(imgScaled)
    #    plt.scatter(source[0, sourceLandmarkInds], source[1, sourceLandmarkInds], s = 1)
    #
    #    plt.title(fName)
    #    if not os.path.exists('landmarkOptPic'):
    #        os.makedirs('landmarkOptPic')
    #    savefig('landmarkOptPic/' + fName + '.png', bbox_inches='tight')
    #    plt.close('all')
    
    # State shape pics
    #view = np.load('view.npz')
    #for i in range(150):
    #    fName = '{:0>5}'.format(i + 1)
    #    shape = importObj('stateShapes/' + fName + '.obj', dataToImport = ['v'])[0].T
    #    
    #    mlab.figure(bgcolor = (1, 1, 1))
    #    tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (0.8, 0.8, 0.8))
    #    mlab.view(view['v0'], view['v1'], view['v2'], view['v3'])
    #    mlab.gcf().scene.parallel_projection = True
    #    
    ##    break
    #    if not os.path.exists('stateShapePics'):
    #        os.makedirs('stateShapePics')
    #    mlab.savefig('stateShapePics/' + fName + '.png', figure = mlab.gcf())
    #    mlab.close(all = True)
    #
    ## Original
    #param = np.load('paramRTS2Orig.npy')
    #view = np.load('viewInFrame.npz')
    #for i in range(numFrames):
    #    fName = '{:0>5}'.format(i + 1)
    #    im = (mpimg.imread('orig/' + fName + '.png') * 255).astype(np.uint8)
    #    mlab_imshowColor(im)
    #    
    #    shape = generateFace(param[i, :], m)
    #    tmesh = mlab.triangular_mesh(shape[0, :]-640, shape[1, :]-360, shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1), opacity = 0.55)
    #    mlab.view(view['v0'], view['v1'], view['v2'], view['v3'])
    #    mlab.gcf().scene.parallel_projection = True
    #    
    #    mlab.gcf().scene.camera.zoom(3.5)
    #    mlab.move(up = 100)
    ##    break
    #    
    #    if not os.path.exists('fitPic'):
    #        os.makedirs('fitPic')
    #    mlab.savefig('fitPic/' + fName + '.png', figure = mlab.gcf())
    #    mlab.close(all = True)
    #
    #param = np.load('paramWithoutRTS.npy')
    #view = np.load('view.npz')
    #for i in range(numFrames):
    #    fName = '{:0>5}'.format(i + 1)
    #    
    #    shape = generateFace(param[i, :], m)
    #    tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1), opacity = 1)
    #    mlab.view(view['v0'], view['v1'], view['v2'], view['v3'])
    #    mlab.gcf().scene.parallel_projection = True
    #    
    #    if not os.path.exists('fitPicWithoutFrame'):
    #        os.makedirs('fitPicWithoutFrame')
    #    mlab.savefig('fitPicWithoutFrame/' + fName + '.png', figure = mlab.gcf())
    #    mlab.close(all = True)
    #
    ## Siro reproduction
    #view = np.load('view.npz')
    #stateSeq_siro = np.load('siroStateSequence.npy')
    #for i in range(numFrames):
    #    fName = '{:0>5}'.format(i + 1)
    #    
    #    shape = importObj('stateShapes/' + '{:0>5}'.format(stateSeq_siro[i] + 1) + '.obj', dataToImport = ['v'])[0].T
    #    tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1), opacity = 1)
    #    mlab.view(view['v0'], view['v1'], view['v2'], view['v3'])
    #    mlab.gcf().scene.parallel_projection = True
    #    
    ##    break
    #    if not os.path.exists('siro'):
    #        os.makedirs('siro')
    #    mlab.savefig('siro/' + fName + '.png', figure = mlab.gcf())
    #    mlab.close(all = True)
    
    #param = np.load('paramRTS2Orig.npy')
    #im = (mpimg.imread('orig/00001.png') * 255).astype(np.uint8)
    #mlab_imshowColor(im)
    #shape = generateFace(param[0, :], m)
    #tmesh = mlab.triangular_mesh(shape[0, :]-640, shape[1, :]-360, shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1), opacity = 0.55)
    #
    #view = mlab.view()
    #mlab.view(180, view[1], view[2], view[3])
    #mlab.gcf().scene.parallel_projection = True
    #np.savez('viewInFrame', v0 = view[0], v1 = view[1], v2 = view[2], v3 = view[3])