import os
import numpy as np
from PIL import Image
import OpenGL
from OpenGL.GL import *
from scipy import misc


def loadVolume(dirName):
    files = sorted(os.listdir(dirName))
    print('loading images from: %s' % dirName)
    imgDatalist = []
    count = 0
    width, height = 0, 0
    for file in files:
        file_path = os.path.abspath(os.path.join(dirName, file))
        try:
            img = Image.open(file_path)
            imgData = np.array(img.getdata(), np.uint8)
            if count is 0:
                width, height = img.size[0], img.size[1]
                imgDatalist.append(imgData)
            else:
                if (width, height) == (img.size[0], img.size[1]):
                    imgDatalist.append(imgData)
                else:
                    print('mismatch')
                    raise RuntimeError("image size mismathc")
            count += 1
        except:
            print('Invalid image: %s' % file_path)
    depth = count
    data = np.concatenate(imgDatalist)
    print('volume data dims: %d %d %d' % (width, height, depth))
    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_3D, texture)
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, width, height, depth, 0, GL_RED, GL_UNSIGNED_BYTE, data)
    return texture, width, height, depth


def loadTexture(filename):
    img = Image.open(filename)
    img_data = np.array(list(img.getdata()), 'B')
    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.size[0], img.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture