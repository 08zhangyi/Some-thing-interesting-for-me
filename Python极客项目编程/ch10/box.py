import sys, random, math
import OpenGL
from OpenGL.GL import *
import numpy
import glutils

strVS = """
#version 330 core

in vec3 aVert;
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
out vec4 vCol;

void main() {
    // apply transformations
    gl_Position = uPMatrix * uMVMatrix * vec4(aVert, 1.0);
    // set color
    vCol = vec4(0.8, 0.0, 0.0, 1.0);
}"""

strFS = """
#version 330 core

in vec4 vCol;
out vec4 fragColor;

void main() {
    // use vertex color
    fragColor = vCol;
}"""


class Box:
    def __init__(self, side):
        self.side = side
        self.program = glutils.loadShaders(strVS, strFS)
        glUseProgram(self.program)

        s = side/2.0
        vertices = [-s, s, -s,
                    -s, -s, -s,
                    s, s, -s,
                    s, -s, -s,
                    s, s, -s,
                    -s, -s, -s,

                    -s, s, s,
                    -s, -s, s,
                    s, s, s,
                    s, -s, s,
                    s, s, s,
                    -s, -s, s,

                    -s, -s, s,
                    -s, -s, -s,
                    s, -s, s,
                    s, -s, -s,
                    s, -s, s,
                    -s, -s, -s,

                    -s, s, s,
                    -s, s, -s,
                    s, s, s,
                    s, s, -s,
                    s, s, s,
                    -s, s, -s,

                    -s, -s, s,
                    -s, -s, -s,
                    -s, s, s,
                    -s, s, -s,
                    -s, s, s,
                    -s, -s, -s,

                    s, -s, s,
                    s, -s, -s,
                    s, s, s,
                    s, s, -s,
                    s, s, s,
                    s, -s, -s,]

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        vertexData = numpy.array(vertices, numpy.float32)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertexData), vertexData, GL_STATIC_DRAW)
        self.vertIndex = glGetAttribLocation(self.program, b"aVert")
        glEnableVertexAttribArray(self.vertIndex)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(self.vertIndex, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def render(self, pMatrix, mvMatrix):
        glUseProgram(self.program)
        glUniformMatrix4fv(glGetUniformLocation(self.program, b"uPMatrix"), 1, GL_FALSE, pMatrix)
        glUniformMatrix4fv(glGetUniformLocation(self.program, b"uMVMatrix"), 1, GL_FALSE, mvMatrix)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)