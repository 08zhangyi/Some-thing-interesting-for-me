import OpenGL
from OpenGL.GL import *
import numpy, math, sys, os

import glutils
import glfw

strVS = """
#version 330 core

layout(location=0) in vec3 aVert;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform float uTheta;

out vec2 vTexCoord;

void main{} {
    // rotation transform
    mat4 rot = mat4(vec4(cos(uTheta), sin(uTheta), 0.0, 0.0),
                    vec4(-sin(uTheta), cos(uTheta), 0.0, 0.0),
                    vec4(0.0, 0.0, 1.0, 0.0),
                    vec4(0.0, 0.0, 0.0, 1.0));
    // transform vertex
    gl_Position = uPMatrix * uMVMatrix * rot * vec4(aVert, 1.0);
    // set texture coordinate
    vTexCoord = aVert.xy + vec2(0.5, 0.5);
}"""

strFS = """
#version 330 core

in vec2 vTextCoord;

uniform sampler2D = tex2D;
uniform bool showCircle;

out vec4 fragColor;

void main() {
    if (showCircle) {
        // discard fragment outside circle
        if (distance(vTexCoord, vec2(0.5, 0.5)) > 0.5) {
            discard;
        } else {
            fragColor = texture(text2D, vTexCoord);
        }
    } else {
        fragColor = texture(tex2D, vTexCoord);
    }
}"""


class Scene:
    def __init__(self):
        self.program = glutils.loadShaders(strVS, strFS)
        glUseProgram(self.program)
        self.pMatrixUniform = glGetUniformLocation(self.program, b'uPMatrix')
        self.mvMatrixUniform = glGetUniformLocation(self.program, b'uMVMatrix')
        self.tex2D = glGetUniformLocation(self.program, b'tex2D')

        vertexData = numpy.array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.0], np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertexData), vertexData, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        self.t = 0
        self.texId = glutils.loadTexture('star.png')
        self.showCircle = False

    def step(self):
        self.t = (self.t+1) % 360
        glUniform1f(glGetUniformLocation(self.program, 'uTheta'), math.radians(self.t))

    def render(self, pMatrix, mvMatrix):
        glUseProgram(self.program)
        glUniformMatrix4fv(self.pMatrixUniform, 1, GL_FALSE, pMatrix)
        glUniformMatrix4fv(self.mvMatrixUniform, 1, GL_FALSE, mvMatrix)
        glUniform1i(glGetUniformLocation(self.program, b'showCircle'), self.showCircle)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texId)
        glUniform1i(self.tex2D, 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


class RenderWindow:
    def __init__(self):
        cwd = os.getcwd()
        glfw.glfwInit()
        os.chdir(cwd)

        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MINOR, 3)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_PROFILE, glfw.GLFW_OPENGL_CORE_PROFILE)

        self.width, self.height = 640, 480
        self.aspect = self.width / float(self.height)
        self.win = glfw.glfwCreateWindow(self.width, self.height, b'simpleglfw')

        glfw.glfwMakeContextCurrent(self.win)

        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.5, 0.5, 0.5, 1.0)

        glfw.glfwSetMouseButtonCallback(self.win, self.onMouseButton)
        glfw.glfwSetKeyCallback(self.win, self.onKeyboard)
        glfw.glfwSetWindowSizeCallback(self.win, self.onSize)

        self.scene = Scene()
        self.exitNow = False

    def onMouseButton(self, win, button, action, mods):
        pass

    def onKeyboard(self, win, key, scancode, action, mods):
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_ESCAPE:
                self.exitNow = True
            else:
                self.scene.showCircle = not self.scene.showCircle

    def onSize(self, win, width, height):
        self.width = width
        self.height = height
        self.aspect = width/float(height)
        glViewport(0, 0, self.width, self.height)

    def run(self):
        glfw.glfwSetTime(0)
        t = 0.0
        while not glfw.glfwWindowShouldClose(self.win) and not self.exitNow:
            currT = glfw.glfwGetTime()
            if currT - t > 0.1:
                t = currT
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                pMatrix = glutils.perspective(45.0, self.aspect, 0.1, 100.0)
                mvMatrix = glutils.lookAt([0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
                self.scene.render(pMatrix, mvMatrix)
                self.scene.step()
                glfw.glfwSwapBuffers(self.win)
                glfw.glfwPollEvents()
        glfw.glfwTerminate()

    def step(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        pMatrix = glutils.perspective(45.0, self.aspect, 0.1, 100.0)
        mvMatrix = glutils.lookAt([0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.scene.render(pMatrix, mvMatrix)
        self.scene.step()
        glfw.glfwSwapBuffers(self.win)
        glfw.glfwPollEvents()


def main():
    print("Starting simpleglfw. \nPress any Key to toggle cut. Press ESC to quit.")
    rw = RenderWindow()
    rw.run()


if __name__ == '__main__':
    main()