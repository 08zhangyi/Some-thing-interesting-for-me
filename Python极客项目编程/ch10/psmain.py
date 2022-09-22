import sys, os, math, numpy
import OpenGL
from OpenGL.GL import *
from ps import ParticleSystem, Camera
from box import Box
import glutils
import glfw


class PSMaker:
    def __init__(self):
        self.camera = Camera([15.0, 0.0, 2.5],
                             [0.0, 0.0, 2.5],
                             [0.0, 0.0, 1.0])
        self.aspect = 1.0
        self.numP = 300
        self.t = 0
        self.rotate = True
        cwd = os.getcwd()
        glfw.glfwInit()
        os.chdir(cwd)

        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MINOR, 3)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_PROFILE, glfw.GLFW_OPENGL_CORE_PROFILE)

        self.width, self.height = 640, 480
        self.aspect = self.width / float(self.height)
        self.win = glfw.glfwCreateWindow(self.width, self.height, b"Particle System")
        glfw.glfwMakeContextCurrent(self.win)

        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glfw.glfwSetMouseButtonCallback(self.win, self.onMouseButton)
        glfw.glfwSetKeyCallback(self.win, self.onKeyboard)
        glfw.glfwSetWindowSizeCallback(self.win, self.onSize)
        self.psys = ParticleSystem(self.numP)
        self.box = Box(1.0)

        self.exitNow = False

    def step(self):
        self.t += 10
        self.psys.step()
        if self.rotate:
            self.camera.rotate()
        if not int(self.t) % 5000:
            self.psys.restart(self.numP)

    def onMouseButton(self, win, button, action, mods):
        pass

    def onKeyboard(self, win ,key, scancode, action, mods):
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_ESCAPE:
                self.exitNow = True
            elif key == glfw.GLFW_KEY_R:
                self.rotate = not self.rotate
            elif key == glfw.GLFW_KEY_B:
                self.psys.enableBillboard = not self.psys.enableBillboard
            elif key == glfw.GLFW_KEY_D:
                self.psys.disableDepthMask = not self.psys.disableDepthMask
            elif key == glfw.GLFW_KEY_T:
                self.psys.enableBlend = not self.psys.enableBlend

    def onSize(self, win, width, height):
        self.width = width
        self.height = height
        self.aspect = width / float(height)
        glViewport(0, 0, self.width, self.height)

    def run(self):
        glfw.glfwSetTime(0)
        t = 0.0
        while not glfw.glfwWindowShouldClose(self.win) and not self.exitNow:
            currT = glfw.glfwGetTime()
            if currT - t > 0.01:
                t = currT
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                pMatrix = glutils.perspective(100.0, self.aspect, 0.1, 100.0)
                mvMatrix = glutils.perspective(self.camera.eye, self.camera.center, self.camera.up)
                self.box.render(pMatrix, mvMatrix)
                self.psys.render(pMatrix, mvMatrix, self.camera)
                self.step()
                glfw.glfwSwapBuffers(self.win)
                glfw.glfwPollEvents()
        glfw.glfwTerminate()


def main():
    print('starting particle system...')
    prog = PSMaker()
    prog.run()


if __name__ == '__main__':
    main()