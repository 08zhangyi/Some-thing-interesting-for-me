import OpenGL
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import numpy as np
import math, sys
import volreader, glutils

strVS = """
#version 330 core

layout(location=1) in vec3 cubePos;
layout(location=2) in vec3 cubeCol;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
out vec4 vColor;

void main() {
    // set back-face color
    vColor = vec4(cubeCol.rgb, 1.0);
    
    // transformed position
    vec4 newPos = vec4(cubePos.xyz, 1.0);
    
    // set position
    gl_Position = uPMatrix*uMVMatrix*newPos;
}"""

strFS = """
#version 330 core

in vec4 vColor;
out  vec4 fragColor;

void main() {
    fragColor = vColor;
}"""


class RayCube:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.program = glutils.loadShaders(strVS, strFS)

        vertices = np.array([0.0, 0.0, 0.0,
                             1.0, 0.0, 0.0,
                             1.0, 1.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0,
                             1.0, 0.0, 1.0,
                             1.0, 1.0, 1.0,
                             0.0, 1.0, 1.0], np.float32)
        colors = np.array([0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0,
                           1.0, 1.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0,
                           1.0, 0.0, 1.0,
                           1.0, 1.0, 1.0,
                           0.0, 1.0, 1.0], np.float32)
        indices = np.array([4, 5, 7,
                            7, 5, 6,
                            5, 1, 6,
                            6, 1, 2,
                            1, 0, 2,
                            2, 0, 3,
                            0, 4, 3,
                            3, 4, 7,
                            6, 2, 7,
                            7, 2, 3,
                            4, 0, 5,
                            5, 0, 1], np.float32)

        self.nIndices = indices.size

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertices), vertices, GL_STATIC_DRAW)
        self.colorBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(colors), colors, GL_STATIC_DRAW)
        self.indexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.indexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 2 * len(indices), indices, GL_STATIC_DRAW)

        aPosLoc = 1
        aColorLoc = 2
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(aPosLoc, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorBuffer)
        glVertexAttribPointer(aColorLoc, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.indexBuffer)

        glBindVertexArray(0)
        self.initFBO()

    def initFBO(self):
        self.fboHandle = glGenFramebuffers(1)
        self.texHandle = glGenTextures(1)
        self.depthHandle = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fboHandle)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texHandle)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texHandle, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depthHandle)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthHandle)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status == GL_FRAMEBUFFER_COMPLETE:
            pass
        elif status == GL_FRAMEBUFFER_UNSUPPORTED:
            print("fbo %d unsupported" % self.fboHandle)
        else:
            print("fbo %d Error" % self.fboHandle)

        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        return

    def closeFBO(self):
        if glIsFramebuffer(self.fboHandle):
            glDeleteFramebuffers(int(self.fboHandle))
        if glDeleteTextures(self.texHandle):
            glDeleteTextures(int(self.texHandle))

    def renderBackFace(self, pMatrix, mvMatrix):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboHandle)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texHandle)

        self.renderCube(pMatrix, mvMatrix, self.program, True)

        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        return self.texHandle

    def renderFrontFace(self, pMatrix, mvMatrix, program):
        self.renderCube(pMatrix, mvMatrix, program, False)

    def renderCube(self, pMatrix, mvMatrix, program, cullFace):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        glUniformMatrix4fv(glGetUniformLocation(program, b'uPMatrix'), 1, GL_FALSE, pMatrix)
        glUniformMatrix4fv(glGetUniformLocation(program, b'uMVMatrix'), 1, GL_FALSE, pMatrix)
        glDisable(GL_CULL_FACE)
        if cullFace:
            glFrontFace(GL_CCW)
            glCullFace(GL_FRONT)
            glEnable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.nIndices, GL_UNSIGNED_SHORT, None)
        glBindVertexArray(0)
        if cullFace:
            glDisable(GL_CULL_FACE)

    def reshape(self, width, height):
        self.width = width
        self.height = height
        self.aspect = width/float(height)
        self.clearFBO()
        self.initFBO()

    def close(self):
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        if glIsFramebuffer(self.fboHandle):
            glDeleteFramebuffers(int(self.fboHandle))
        if glIsTexture(self.texHandle):
            glDeleteTextures(int(self.texHandle))
        if glIsRenderbuffer(self.depthHandle):
            glDeleteRenderbuffers(1, int(self.depthHandle))
