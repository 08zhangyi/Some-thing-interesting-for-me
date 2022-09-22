import sys, argparse, os
from slicerender import *
from raycast import *
import glfw


class RenderWin:
    def __init__(self, imageDir):
        cwd = os.getcwd()
        glfw.glfwInit()
        os.chdir(cwd)
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MINOR, 3)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_PROFILE, glfw.GLFW_OPENGL_CORE_PROFILE)

        self.width, self.height = 512, 512
        self.aspect = self.width/float(self.height)
        self.win = glfw.glfwCreateWindow(self.width, self.height, b"volrender")
        glfw.glfwMakeContextCurrent(self.win)
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glfw.glfwSetMouseButtonCallback(self.win, self.onMouseButton)
        glfw.glfwSetKeyCallback(self.win, self.onKeyboard)
        glfw.glfwSetWindowSizeCallback(self.win ,self.onSize)

        self.volume = volreader.loadVolume(imageDir)
        self.render = RayCastRender(self.width, self.height, self.volume)

        self.exitNow = False

    def onMouseButton(self, win, button, action, mods):
        pass

    def onKeyboard(self, win, key, scancode, action, mods):
        if key is glfw.GLFW_KEY_ESCAPE:
            self.render.close()
            self.exitNow = True
        else:
            if action is glfw.GLFW_PRESS or action is glfw.GLFW_REPEAT:
                if key == glfw.GLFW_KEY_V:
                    if isinstance(self.render, RayCastRender):
                        self.renderer = SliceRender(self.width, self.height, self.volume)
                    else:
                        self.renderer = RayCastRender(self.width, self.height, self.volume)
                    self.renderer.reshape(self.width, self.height)
                else:
                    keyDict = {glfw.GLFW_KEY_X: 'x', glfw.GLFW_KEY_Y: 'y', glfw.GLFW_KEY_Z: 'z',
                               glfw.GLFW_KEY_LEFT: 'l', glfw.GLFW_KEY_RIGHT: 'r'}
                    try:
                        self.renderer.keyPressed(keyDict[key])
                    except:
                        pass

    def onSize(self, win, width, height):
        self.width = width
        self.height = height
        self.aspect = width/float(height)
        glViewport(0, 0, self.width, self.height)
        self.renderer.reshape(width, height)

    def run(self):
        while not glfw.glfwWindowShouldClose(self.win) and not self.exitNow:
            self.renderer.draw()
            glfw.glfwSwapBuffers(self.win)
            glfw.glfwWaitEvents()
        glfw.glfwTerminate()


def main():
    print('starting volrender...')
    parser = argparse.ArgumentParser(description="Volume Rendering...")
    parser.add_argument('--dir', dest='imageDir', required=True)
    args = parser.parse_args()
    rwin = RenderWin(args.imageDir)
    rwin.run()


if __name__ == '__main__':
    main()