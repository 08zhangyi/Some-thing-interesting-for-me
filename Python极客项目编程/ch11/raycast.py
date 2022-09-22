import OpenGL
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import numpy as np
import math, sys
import raycube, glutils, volreader

strVS = """
#version 330 core

layout(location=1) in vec3 cubePos;
layout(location=2) in vec3 cubeCol;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;

out vec4 vColor;

void main() {
    // set position
    gl_position = uPMatrix * uMVMatrix * vec4(cubePos.xyz, 1.0);
    // set color
    vColor = vec4(cubeCol.rgb, 1.0);
}"""

strFS = """
#version 330 core

in vec4 vColor;

uniform sampler2D texBackFaces;
uniform sampler3D texVolume;
uniform vec2 uWinDims;

out vec4 fragColor;

void main() {
    // start of ray
    vec3 start = vColor.rgb;
    // calculate texture coords at fragment, which is a fraction of window coords
    vec2 texc = gl_FragCoord.xy/uWinDims.xy;
    // get end of ray by looking up back-face color
    vec3 end = texture(texBackFaces, texc).rgb;
    // calculate ray direction
    vec3 dir = end - start;
    // normalized ray direction
    vec3 norm_dir = normalize(dir);
    // the length from front to back is calculated and used to terminate the ray
    float len = length(dir.xyz);
    // ray step size
    float stepSize = 0.01;
    // x-ray projection
    vec4 dst = vec4(0.0);
    // step through the ray
    for (float t=0.0; t<len; t+=stepSize) {
        // set position to end point of ray
        vec3 samplePos = start + t*norm_dir;
        // get texture value at position
        float val = texture(texVolume, samplePos).r;
        vec4 src = vec4(val);
        // set opacity
        src.a *= 0.1;
        src.rgb *= src.a;
        // blend with previous value
        dst = (1.0 - dst.a)*src + dst;
        // exit loop when alpha exceeds threshold
        if (dst.a >= 0.95)
            break;
    }
    // set fragment color
    fragColor = dst;
}"""


class Camera:
    def __init__(self):
        self.r = 1.5
        self.theta = 0
        self.center = [0.5, 0.5, 0.5]
        self.eye = [0.5+self.r, 0.5, 0.5]
        self.up = [0.0, 0.0, 1.0]

    def rotate(self, clockWise):
        if clockWise:
            self.theta = (self.theta + 5) % 360
        else:
            self.theta = (self.theta - 5) % 360
        self.eye = [0.5+self.r*math.cos(math.radians(self.theta)),
                    0.5+self.r*math.sin(math.radians(self.theta)),
                    0.5]


class RayCastRender:
    def __init__(self, width, height, volume):
        self.raycube = raycube.RayCube(width, height)
        self.width = width
        self.height = height
        self.aspect = width/float(height)
        self.program = glutils.loadShaders(strVS, strFS)
        self.texVolume, self.Nx, self.Ny, self.Nz = volume
        self.camera = Camera()

    def draw(self):
        pMatrix = glutils.perspective(45.0, self.aspect, 0.1, 100.0)
        mvMatrix = glutils.lookAt(self.camera.eye, self.camera.center, self.camera.up)
        texture = self.raycube.renderBackFace(pMatrix, mvMatrix)
        glUseProgram(self.program)

        glUniform2f(glGetUniformLocation(self.program, b"uWinDims"), float(self.width), float(self.height))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(glGetUniformLocation(self.program, b"texBackFaces"), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, texture)
        glUniform1i(glGetUniformLocation(self.program, b"texVolume"), 1)
        self.raycube.renderFrontFace(pMatrix, mvMatrix, self.program)

    def keyPressed(self, key):
        if key == 'l':
            self.camera.rotate(True)
        elif key == 'r':
            self.camera.rotate(False)

    def reshape(self, width, height):
        self.width = width
        self.height = height
        self.aspect = width/float(height)
        self.raycube.reshape(width, height)

    def close(self):
        self.raycube.close()