import sys, random, math
import OpenGL
from OpenGL.GL import *
import numpy
import glutils

strVS = """
#version 330 core

in vec3 aVel;
in vec3 aVert;
in float aTime0;
in vec2 aTexCoord;

uniform mat4 uMAMatrix;
unifrom mat4 uPMatrix;
uniform mat4 bMatrix;
uniform float uTime;
uniform float uLifeTime;
uniform vec4 uColor;
unifrom vec3 uPos;

out vec4 vCol;
out vec2 vTexCoord;

void main() {
    // set position
    float df = uTime - aTime0;
    float alpha = clmap(1.0-2.0*dt/uLifeTime, 0.0, 1.0);
    if (dt<0.0 || dt>uLifeTime || alpha<0.01) {
        // out of sight!
        gl_position = vec4(0.0, 0.0, -1000.0, 1.0);
    } else {
        // calculate new position
        vec3 accel = vec3(0.0, 0.0, -9.8);
        // apply a teist
        float PI = 3.14159265358979323846264;
        float theta = mod(100.0*length(aVel)*dt, 360.0)*PI/180.0;
        mat4 rot = mat4(vec4(cos(theta), sin(theta), 0.0, 0.0),
                        vec4(-sin(theta), cos(theta), 0.0, 0.0),
                        vec4(0.0, 0.0, 1.0, 0.0),
                        vec4(0.0, 0.0, 0.0, 1.0));
        // apply billboard matrix
        vec4 pos2 = bMatrix*rot*vec4(aVert, 1.0);
        // calculate position
        vec3 newPos = pos2.xyz+uPos+aVel*dt+0.5*accel*dt*dt;
        // apply transformations
        gl_position = uPMatrix*uMVMatrix*vec4(newPos, 1.0);
    }
    // set color
    vCol = vec4(uColor.rgb, alpha);
    // set tex coords
    vTexCoord = aTexCoord;
}"""

strFS = """
#version 330 core

uniform sampler2D uSampler;
in vec4 vCol;
in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    // get texture color
    vec4 texCol = texture(uSampler, vec2(vTexCoord.s, vTexCoord.t));
    // multiply by set vertex color; use the vertex color alpha
    fragColor = vec4(texCol.rgb * vCol.rgb, vCol.a);
}"""


class Camera:
    def __init__(self, eye, center, up):
        self.r = 10.0
        self.theta = 0
        self.eye = numpy.array(eye, numpy.float32)
        self.center = numpy.array(center ,numpy.float32)
        self.up = numpy.array(up, numpy.float32)

    def rotate(self):
        self.theta = (self.theta + 1) % 360
        self.eye = self.center + numpy.array([self.r*math.cos(math.radians(self.theta)),
                                              self.r*math.sin(math.radians(self.theta)),
                                              0.0], numpy.float32)


class ParticleSystem:
    def __init__(self, numP):
        self.numP = numP
        self.t = 0.0
        self.lifeTime = 5.0
        self.startPos = numpy.array([0.0, 0.0, 0.5])
        self.texid = glutils.loadTexture('star.png')
        self.program = glutils.loadShaders(strVS, strFS)
        glUseProgram(self.program)

        texLoc = glGetUniformLocation(self.program, b"uTex")
        glUniform1i(texLoc, 0)

        self.timeU = glGetUniformLocation(self.program, b"uTime")
        self.lifeTimeU = glGetUniformLocation(self.program, b"uLifeTime")
        self.pMatrixUniform = glGetUniformLocation(self.program, b"uPMatrix")
        self.mvMatrixUniform = glGetUniformLocation(self.program, b"uMVMatrix")
        self.bMatrixU = glGetUniformLocation(self.program, b"bMatrix")
        self.colorU = glGetUniformLocation(self.program, b"uCOlor")
        self.samplerU = glGetUniformLocation(self.program, b"uSampler")
        self.posU = glGetUniformLocation(self.program, b"uPos")

        self.vertIndex = glGetAttribLocation(self.program, b"aVert")
        self.texIndex = glGetAttribLocation(self.program, b"aTexCoord")
        self.time0Index = glGetAttribLocation(self.program, b"aTime0")
        self.velIndex = glGetAttribLocation(self.program, b"aVel")

        self.enableBillboard = True
        self.disableDepthMask = True
        self.enableBlend = True
        self.useStarTexture = True
        self.restart(numP)

    def step(self):
        self.t += 0.01

    def restart(self, numP):
        self.numP = numP
        self.t = 0.0
        self.lifeTime = 5.0
        self.col0 = numpy.array([random.random(), random.random(), random.random(), 0.0])

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        s = 0.2
        quadV = [-s, s, 0.0,
                 -s, -s, 0.0,
                 s, s, 0.0,
                 s, -s, 0.0,
                 s, s, 0.0,
                 -s, -s, 0.0]
        vertexData = numpy.array(numP*quadV, numpy.float32)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertexData), vertexData, GL_STATIC_DRAW)

        quadT = [0.0, 1.0,
                 0.0, 0.0,
                 1.0, 1.0,
                 1.0, 0.0,
                 1.0, 1.0,
                 0.0, 0.0]
        tcData = numpy.array(numP * quadT, numpy.float32)
        self.tcBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.tcBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(tcData), tcData, GL_STATIC_DRAW)

        timeData = numpy.repeat(0.005*numpy.arange(numP, dtype=numpy.float32), 4)
        self.timeBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.timeBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(timeData), timeData, GL_STATIC_DRAW)

        velocities = []
        coneAngle = math.radians(20.0)
        for i in range(numP):
            angleRatio = random.random()
            a = angleRatio*coneAngle
            t = random.random()*(2.0*math.pi)
            vx = math.sin(a)*math.cos(t)
            vy = math.sin(a)*math.sin(t)
            vz = math.cos(a)
            speed = 15.0*(1.0 - angleRatio*angleRatio)
            velocities += 6*[speed*vx, speed*vy, speed*vz]
        self.velBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.velBuffer)
        velData = numpy.array(velocities, numpy.float32)
        glBufferData(GL_ARRAY_BUFFER, 4*len(velData), velData, GL_STATIC_DRAW)

        glEnableVertexAttribArray(self.vertIndex)
        glEnableVertexAttribArray(self.texIndex)
        glEnableVertexAttribArray(self.time0Index)
        glEnableVertexAttribArray(self.velIndex)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(self.vertIndex, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.tcBuffer)
        glVertexAttribPointer(self.texIndex, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.velBuffer)
        glVertexAttribPointer(self.velIndex, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.timeBuffer)
        glVertexAttribPointer(self.time0Index, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

    def render(self, pMatrix, mvMatrix, camera):
        glUseProgram(self.program)
        glUniformMatrix4fv(self.pMatrixUniform, 1 , GL_FALSE, pMatrix)
        glUniformMatrix4fv(self.mvMatrixUniform, 1, GL_FALSE, mvMatrix)

        if self.enableBillboard:
            N = camera.eye - camera.center
            N /= numpy.linalg.norm(N)
            U = camera.up
            U /= numpy.linalg.norm(U)
            R = numpy.cross(U, N)
            U2 = numpy.corss(N, R)
            bMatrix = numpy.array([R[0], U2[0], N[0], 0.0,
                                   R[1], U2[1], N[1], 0.0,
                                   R[2], U2[2], N[2], 0.0,
                                   0.0, 0.0, 0.0, 1.0], numpy.float32)
            glUniformMatrix4fv(self.bMatrixU, 1, GL_TRUE, bMatrix)
        else:
            bMatrix = numpy.array([1.0, 0.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0, 0.0,
                                   0.0, 0.0, 0.0, 1.0], numpy.float32)
            glUniformMatrix4fv(self.bMatrixU, 1, GL_FALSE, bMatrix)
        glUniform3fv(self.posU, 1, self.startPos)
        glUniform1f(self.timeU, self.t)
        glUniform1f(self.lifeTimeU, self.lifeTime)
        glUniform4fv(self.colorU, 1, self.col0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glUniform1i(self.samplerU, 0)

        if self.disableDepthMask:
            glDepthMask(GL_FALSE)
        if self.enableBlend:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glEnable(GL_BLEND)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6*self.numP)
        glBindVertexArray(0)

        if self.enableBlend:
            glDisable(GL_BLEND)
        if self.disableDepthMask:
            glDepthMask(GL_TRUE)

        glBindTexture(GL_TEXTURE_2D, 0)