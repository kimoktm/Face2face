#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np

def windowToClip(width, height, zNear, zFar):
    """Creates elements for an OpenGL-style column-based orthographic transformation matrix that maps homogenous coordinates from window space to clip space.
    
    Args:
        width (int): Pixel width of window/viewport
        height (int): Pixel height of window/viewport
        zNear (int): Nearside clipping plane
        zFar (int): Farside clipping plane
    
    Returns:
        ndarray, (16,): Vectorized transformation matrix
    """
    windowToClipMat = np.zeros(16, dtype = np.float32)
    windowToClipMat[0] = 2 / width
    windowToClipMat[3] = -1
    windowToClipMat[5] = 2 / height
    windowToClipMat[7] = -1
    windowToClipMat[10] = 2 / (zFar - zNear)
    windowToClipMat[11] = -(zFar + zNear) / (zFar - zNear)
    windowToClipMat[15] = 1
    
    return windowToClipMat

# Strings containing shader programs written in GLSL
vertexShaderString = """
#version 330

layout(location = 0) in vec3 windowCoordinates;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 barycentricCoordinates;
layout(location = 3) in uint triangleID;

smooth out vec3 fragmentColor;
smooth out vec3 fragmentBarycentricCoordinates;
flat out uint fragmentFaceID;

uniform mat4 windowToClipMat;

void main()
{
    gl_Position = windowToClipMat * vec4(windowCoordinates, 1.0f);
    fragmentColor = vertexColor;
    fragmentBarycentricCoordinates = barycentricCoordinates;
    fragmentFaceID = triangleID;
}
"""
"""str: OpenGL GLSL vertex shader
"""

fragmentShaderString = """
#version 330

smooth in vec3 fragmentColor;
smooth in vec3 fragmentBarycentricCoordinates;
flat in uint fragmentFaceID;

layout(location = 0) out vec4 pixelColor;
layout(location = 1) out vec3 pixelBarycentricCoordinates;
layout(location = 2) out uvec3 pixelFaceID;

void main()
{
    pixelColor = vec4(fragmentColor, 1.);
    pixelBarycentricCoordinates = fragmentBarycentricCoordinates;
    pixelFaceID = uvec3(fragmentFaceID, 0, 0);
}
"""
""" str: OpenGL GLSL fragment shader
"""

class Render:
    """OpenGL rendering class
    
    Args:
        width (int): Pixel width of window/viewport
        height (int): Pixel height of window/viewport
        meshData (ndarray): 3DMM vertex coordinates and RGB values, concatenated vertically
        indexData (ndarray): 3DMM vertex indices for each triangular face
        indexed (bool): Determines whether or not to do indexed OpenGL drawing
        img (ndarray, (height, width, 1 or 3)): Optional background image for rendering
            
    Attributes:
        width (int): Pixel width of window/viewport
        height (int): Pixel height of window/viewport
        zNear (int): Nearside clipping plane
        zFar (int): Farside clipping plane
        meshData (ndarray): 3DMM vertex coordinates and RGB values, concatenated vertically
        indexData (ndarray): 3DMM vertex indices for each triangular face
        numVertices (int): Number of 3DMM vertices
        numFaces (int): Number of 3DMM triangular faces
        vertexDim (int): Dimensionality of 3DMM vertices
        indexed (bool): Determines whether or not to do indexed OpenGL drawing
        img (ndarray, (height, width, 1 or 3)): Optional background image for rendering
        window (int): Instance of OpenGL GLUT context
        shaderDict (dict): Dictionary of OpenGL shader strings
        numStoredVertices (int): Number of 3DMM vertices stored in the VBO
    """
    def __init__(self, width, height, meshData, indexData, indexed = False, img = None):
        # Initialize input
        self.width = width
        self.height = height
        self.zNear = -1000
        self.zFar = 1000
        
        self.meshData = meshData.astype(np.float32)
        self.indexData = indexData.astype(np.uint16)
        
        self.numVertices = self.meshData.shape[0] // 2
        self.numFaces = self.indexData.shape[0]
        self.vertexDim = 3
        
        self.indexed = indexed
        self.img = img
        
        self.initializeContext()
    
    def initializeContext(self):
        """Intializes an OpenGL context for rendering.
        """
        # You can use any means to initialize an OpenGL context (e.g. GLUT), but since we're rendering offscreen to an FBO, we don't need to bother to display, which is why we hide the GLUT window.
        glutInit()

        self.window = glutCreateWindow('Merely creating an OpenGL context...')
        glutHideWindow()
        
        # Organize the strings defining our shaders into a dictionary
        self.shaderDict = {GL_VERTEX_SHADER: vertexShaderString, GL_FRAGMENT_SHADER: fragmentShaderString}
        
        # Use this dictionary to compile the shaders and link them to the GPU processors
        self.initializeShaders()
        
        # A utility function to modify uniform inputs to the shaders
        self.configureShaders()
        
        # Set the dimensions of the viewport
        glViewport(0, 0, self.width, self.height)
        
        # Performs face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CW)
        
        # Performs z-buffer testing
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)
        glDepthRange(0.0, 1.0)
        
        # Initialize OpenGL objects
        if self.indexed:
            self.numStoredVertices = self.numVertices
            self.initializeVertexBuffer()
            self.initializeVertexArray()
        else:
            self.numStoredVertices = self.vertexDim * self.numFaces
            barycentricCoord = np.tile(np.eye(3, dtype = np.float32), (self.numFaces, 1))
            self.meshData = np.r_[self.meshData[:self.numVertices, :][self.indexData.flat], self.meshData[self.numVertices:, :][self.indexData.flat], barycentricCoord].astype(np.float32)
            faceID = np.repeat(np.arange(self.numFaces, dtype = np.uint16), 3) + 1
            
            self.initializeVertexBuffer(faceID)
            self.initializeVertexArray()
        
        self.initializeFramebufferObject()

    def initializeShaders(self):
        """Compiles each shader defined in shaderDict, attaches them to a program object, and links them (i.e., creates executables that will be run on the vertex, geometry, and fragment processors on the GPU). This is more-or-less boilerplate.
        """
        shaderObjects = []
        self.shaderProgram = glCreateProgram()
        
        for shaderType, shaderString in self.shaderDict.items():
            shaderObjects.append(glCreateShader(shaderType))
            glShaderSource(shaderObjects[-1], shaderString)
            
            glCompileShader(shaderObjects[-1])
            status = glGetShaderiv(shaderObjects[-1], GL_COMPILE_STATUS)
            if status == GL_FALSE:
                if shaderType is GL_VERTEX_SHADER:
                    strShaderType = "vertex"
                elif shaderType is GL_GEOMETRY_SHADER:
                    strShaderType = "geometry"
                elif shaderType is GL_FRAGMENT_SHADER:
                    strShaderType = "fragment"
                raise RuntimeError("Compilation failure (" + strShaderType + " shader):\n" + glGetShaderInfoLog(shaderObjects[-1]).decode('utf-8'))
            
            glAttachShader(self.shaderProgram, shaderObjects[-1])
        
        glLinkProgram(self.shaderProgram)
        status = glGetProgramiv(self.shaderProgram, GL_LINK_STATUS)
        
        if status == GL_FALSE:
            raise RuntimeError("Link failure:\n" + glGetProgramInfoLog(self.shaderProgram).decode('utf-8'))
            
        for shader in shaderObjects:
            glDetachShader(self.shaderProgram, shader)
            glDeleteShader(shader)
    
    def configureShaders(self):
        """Modifies the window-to-clip space transform matrix in the vertex shader, but you can use this to configure your shaders however you'd like, of course.
        """
        # Grabs the handle for the uniform input from the shader
        windowToClipMatUnif = glGetUniformLocation(self.shaderProgram, "windowToClipMat")
        
        # Get input parameters to define a matrix that will be used as the uniform input
        windowToClipMat = windowToClip(self.width, self.height, self.zNear, self.zFar)
        
        # Assign the matrix to the uniform input in our shader program. Note that GL_TRUE here transposes the matrix because of OpenGL conventions
        glUseProgram(self.shaderProgram)
        glUniformMatrix4fv(windowToClipMatUnif, 1, GL_TRUE, windowToClipMat)
        
        # We're finished modifying our shader, so we set the program to be used as null to be proper
        glUseProgram(0)
    
    def initializeVertexBuffer(self, faceID = None):
        """Assigns the triangular mesh data and the triplets of vertex indices that form the triangles (index data) to VBOs
        
        Args:
            faceID (ndarray): Optional, an array of triangular face ID numbers for each vertex. This is only used with non-indexed OpenGL drawing, where vertices are replicated in the VBO so that each set of three vertices corresponding to a triangular face can have a vertex attribute representing the index number of the triangular face.
        """
        # Create a handle and assign the VBO for the mesh data to it
        self.shaderProgram = glGenBuffers(1)
        
        # Bind the VBO to the GL_ARRAY_BUFFER target in the OpenGL context
        glBindBuffer(GL_ARRAY_BUFFER, self.shaderProgram)
        
        # Allocate enough memory for this VBO to contain the mesh data
        glBufferData(GL_ARRAY_BUFFER, self.meshData, GL_STATIC_DRAW)
        
        # Unbind the VBO from the target to be proper
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        # Similar to the above, but for the vertex index data
        self.indexBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indexData, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        
        if faceID is not None:
            # Similar to the above, but for the face index data
            self.faceIDBufferObject = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.faceIDBufferObject)
            glBufferData(GL_ARRAY_BUFFER, faceID, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def updateVertexBuffer(self, meshData):
        """Updates the VBO with new 3DMM vertex coordinates and RGB colors without having to reinitialize the VBO.
        
        Args:
            meshData (ndarray): 3DMM vertex coordinates and RGB values, concatenated vertically
        """
        # Set the VAO as the currently used object in the OpenGL context
        glBindVertexArray(self.vertexArrayObject)
        
        # Bind the VBO to the GL_ARRAY_BUFFER target in the OpenGL context
        glBindBuffer(GL_ARRAY_BUFFER, self.shaderProgram)
        
        # Replace the mesh data (vertex coordinates and colors) in the VBO
        meshData = meshData.astype(np.float32)
        if not self.indexed:
            meshData = np.r_[meshData[:self.numVertices, :][self.indexData.flat], meshData[self.numVertices:, :][self.indexData.flat]].astype(np.float32)
        
        glBufferSubData(GL_ARRAY_BUFFER, 0, meshData.nbytes, meshData)
        
        # Unbind the VBO from the target to be proper
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        # Unset the VAO as the current object in the OpenGL context
        glBindVertexArray(0)
    
    def initializeFramebufferObject(self):
        """Creates an FBO and assign a texture to it for the purpose of offscreen rendering. Also assigns textures to hold the barycentric coordinates and face IDs for each pixel during the rendering.
        """
        # Create a handle and assign a texture buffer to it
        renderedTexture = glGenTextures(1)
        
        # Bind the texture buffer to the GL_TEXTURE_2D target in the OpenGL context
        glBindTexture(GL_TEXTURE_2D, renderedTexture)
        
        # Attach a texture 'img' (which should be of unsigned bytes) to the texture buffer. If you don't want a specific texture, you can just replace 'img' with 'None'.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, self.img)
        
        # Does some filtering on the texture in the texture buffer
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
        # Unbind the texture buffer from the GL_TEXTURE_2D target in the OpenGL context
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Make a similar texture for the barycentric coordinates of each pixel
        barycentricTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, barycentricTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Make a similar texture for the triangle face ID of each pixel
        faceIDTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, faceIDTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, self.width, self.height, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Create a handle and assign a renderbuffer to it
        depthRenderbuffer = glGenRenderbuffers(1)
        
        # Bind the renderbuffer to the GL_RENDERBUFFER target in the OpenGL context
        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer)
        
        # Allocate enough memory for the renderbuffer to hold depth values for the texture
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        
        # Unbind the renderbuffer from the GL_RENDERBUFFER target in the OpenGL context
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        
        # Create a handle and assign the FBO to it
        self.framebufferObject = glGenFramebuffers(1)
        
        # Use our initialized FBO instead of the default GLUT framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        
        # Attaches the texture buffer created above to the GL_COLOR_ATTACHMENT0 attachment point of the FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderedTexture, 0)
        
        # Attaches the barycentric buffer created above to the GL_COLOR_ATTACHMENT1 attachment point of the FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, barycentricTexture, 0)
        
        # Attaches the faceID buffer created above to the GL_COLOR_ATTACHMENT2 attachment point of the FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, faceIDTexture, 0)
        
        # Attaches the renderbuffer created above to the GL_DEPTH_ATTACHMENT attachment point of the FBO
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer)
        
        # Defines which buffers the fragment shader will draw to
        glDrawBuffers(3, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2])
        
        # Sees if your GPU can handle the FBO configuration defined above
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer binding failed, probably because your GPU does not support this FBO configuration.')
        
        # Unbind the FBO, relinquishing the GL_FRAMEBUFFER back to the window manager (i.e. GLUT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def resetFramebufferObject(self):
        """Erases any drawn objects from the FBO.
        """
        # Use our initialized FBO instead of the default GLUT framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        
        # Clears any color or depth information in the FBO
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Unbind the FBO, relinquishing the GL_FRAMEBUFFER back to the window manager (i.e. GLUT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def initializeVertexArray(self):
        """Creates the VAO to store the VBOs for the mesh data and the index data and assigns the vertex attributes for the OpenGL shaders.
        """
        # Create a handle and assign a VAO to it
        self.vertexArrayObject = glGenVertexArrays(1)
        
        # Set the VAO as the currently used object in the OpenGL context
        glBindVertexArray(self.vertexArrayObject)
        
        # Bind the VBO for the mesh data to the GL_ARRAY_BUFFER target in the OpenGL context
        glBindBuffer(GL_ARRAY_BUFFER, self.shaderProgram)
        
        # Specify the location indices for the types of inputs to the shaders
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        
        # Assign the first type of input (which is stored in the VBO beginning at the offset in the fifth argument) to the shaders
        glVertexAttribPointer(0, self.vertexDim, GL_FLOAT, GL_FALSE, 0, None)
        
        # Calculate the offset in the VBO for the second type of input, specified in bytes (because we used np.float32, each element is 4 bytes)
        colorOffset = c_void_p(self.vertexDim * self.numStoredVertices * 4)
        
        # Assign the second type of input, beginning at the offset calculated above, to the shaders
        glVertexAttribPointer(1, self.vertexDim, GL_FLOAT, GL_FALSE, 0, colorOffset)
        
        if self.indexed:
            # Bind the VBO for the index data to the GL_ELEMENT_ARRAY_BUFFER target in the OpenGL context
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        else:
            # Assign the barycentric coordinates for each mesh triangle vertex as the third input to the shaders
            glEnableVertexAttribArray(2)
            indexOffset = c_void_p(self.vertexDim * self.numStoredVertices * 4 * 2)
            glVertexAttribPointer(2, self.vertexDim, GL_FLOAT, GL_FALSE, 0, indexOffset)
            
            # Assign the face indices for each mesh triangle vertex as the fourth input to the shaders
            glBindBuffer(GL_ARRAY_BUFFER, self.faceIDBufferObject)
            glEnableVertexAttribArray(3)
            glVertexAttribIPointer(3, 1, GL_UNSIGNED_SHORT, 0, None)
        
        # Unset the VAO as the current object in the OpenGL context
        glBindVertexArray(0)
        
    def render(self):
        """Renders the objects defined in the VAO to the FBO.
        """
        # Defines what shaders to use
        glUseProgram(self.shaderProgram)
        
        # Use our initialized FBO instead of the default GLUT framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        
        # Set our initialized VAO as the currently used object in the OpenGL context
        glBindVertexArray(self.vertexArrayObject)
        
        if self.indexed:
            # Draws the mesh triangles defined in the VBO of the VAO above according to the vertices defining the triangles in indexData, which is of unsigned shorts
            glDrawElements(GL_TRIANGLES, self.vertexDim * self.numFaces, GL_UNSIGNED_SHORT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, self.vertexDim * self.numFaces)
        
        # Being proper
        glBindVertexArray(0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def grabRendering(self, return_info = False):
        """Reads the rendered pixels from the FBO into an array.
        
        Args:
            return_info (bool): Optional, can choose whether or not to return the barycentric coordinates and triangular face IDs for each pixel. You can only get this extra information when doing non-indexed OpenGL drawing. If ``False``, it only returns the rendering. If ``True``, it returns a tuple containing the rendering, the pixel coordinates where the 3DMM is drawn, the triangular face IDs for each pixel where the 3DMM is drawn, and the barycentric coordinates of the triangular face underlying each pixel where the 3DMM is drawn.
            
        Returns:
            ndarray or tuple
        """
        # Use our initialized FBO instead of the default GLUT framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        
        # Configure how we store the pixels in memory for our subsequent reading of the FBO to store the rendering into memory. The second argument specifies that our pixels will be in bytes.
    #    glPixelStorei(GL_PACK_ALIGNMENT, 1)
        
        # Set the color buffer that we want to read from. The GL_COLORATTACHMENT0 target is where we assigned our texture buffer in the FBO.
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        
        # Now we do the actual reading, noting that we read the pixels in the buffer as unsigned bytes to be consistent will how they are stored
        rendering = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_FLOAT)
        rendering = np.frombuffer(rendering, dtype = np.float32).reshape(self.height, self.width, 3)
        
        if not return_info:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return rendering
        else:
            glReadBuffer(GL_COLOR_ATTACHMENT1)
            barycentricCoords = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_FLOAT)
            barycentricCoords = np.frombuffer(barycentricCoords, dtype = np.float32).reshape(self.height, self.width, 3)
            
            glReadBuffer(GL_COLOR_ATTACHMENT2)
            faceID = glReadPixels(0, 0, self.width, self.height, GL_RED_INTEGER, GL_UNSIGNED_SHORT)
            faceID = np.frombuffer(faceID, dtype = np.uint16).reshape(self.height, self.width)
            
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            pixelCoord = np.transpose(np.nonzero(faceID))
            pixelFaces = faceID[faceID != 0] - 1
            pixelBarycentricCoords = barycentricCoords[faceID != 0]
            
            return rendering, pixelCoord, pixelFaces, pixelBarycentricCoords