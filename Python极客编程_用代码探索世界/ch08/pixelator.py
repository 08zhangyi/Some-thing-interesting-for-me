import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

files = ['earth_west.png', 'earth_east.png']

for file in files:
    img_ini = cv.imread(file)
    pixelated = cv.resize(img_ini, (3, 3), interpolation=cv.INTER_AREA)
    img = cv.resize(pixelated, (300, 300), interpolation=cv.INTER_AREA)
    cv.imshow('Pixelated {}'.format(file), img)
    cv.waitKey(2000)

    b, g, r = cv.split(pixelated)
    color_aves = []
    for array in (b, g, r):
        color_aves.append(np.average(array))

    labels = 'Blue', 'Green', 'Red'
    colors = ['blue', 'green', 'red']
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    _, _, autotexts = ax.pie(color_aves, labels=labels, autopct='%1.1f%%', colors=colors)
    for autotext in autotexts:
        autotext.set_color('white')
    plt.title('{}\n'.format(file))

plt.show()