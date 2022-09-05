import sys, os, random, argparse
from PIL import Image
import imghdr
import numpy as np


def getImageFilenames(imageDir):
    files = os.listdir(imageDir)
    filenames = []
    for file in files:
        filePath = os.path.abspath(os.path.join(imageDir, file))
        try:
            imgType = imghdr.what(filePath)
            if imgType:
                filenames.append(filePath)
        except:
            print("Invalid image: %s" % (filePath,))
    return filenames


def getImages(imageDir):
    files = os.listdir(imageDir)
    images = []
    for file in files:
        filePath = os.path.abspath(os.path.join(imageDir, file))
        try:
            fp = open(filePath, "rb")
            im = Image.open(fp)
            images.append(im)
            im.load()
            fp.close()
        except:
            print("Invalid image: %s" % (filePath,))
    return images


def getAverageRGB(image):
    im = np.array(image)
    w, h, d = im.shape
    return tuple(np.average(im.reshape(w*h, d), axis=0))


def splitImage(image, size):
    W, H = image.size[0], image.size[1]
    m, n = size
    w, h = int(W/n), int(H/m)
    imgs = []
    for j in range(m):
        for i in range(n):
            imgs.append(image.crop((i*w, j*h, (i+1)*w, (j+1)*h)))
    return imgs


def getBestMatchIndex(input_avg, avgs):
    avg = input_avg
    index = 0
    min_index = 0
    min_dist = float("inf")
    for val in avgs:
        dist = ((val[0]-avg[0])*(val[0]-avg[0])+(val[1]-avg[1])*(val[1]-avg[1])+(val[2]-avg[2])*(val[2]-avg[2]))
        if dist < min_dist:
            min_dist = dist
            min_index = index
        index += 1
    return min_index


def createImageGrid(images, dims):
    m, n = dims
    assert m*n == len(images)
    width = max([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    grid_img = Image.new('RGB', (n*width, m*height))
    for index in range(len(images)):
        row = int(index/n)
        col = index - n*row
        grid_img.paste(images[index], (col*width, row*height))
    return grid_img


def createPhotomosaic(target_image, input_images, grid_size, reuse_images=True):
    print('splitting input image...')
    target_images = splitImage(target_image, grid_size)
    print('finding image matches...')
    output_images = []
    count = 0
    batch_size = int(len(target_images)/10)
    avgs = []
    for img in input_images:
        avgs.append(getAverageRGB(img))
    for img in target_images:
        avg = getAverageRGB(img)
        match_index = getBestMatchIndex(avg, avgs)
        output_images.append(input_images[match_index])
        if count>0 and batch_size>10 and count%batch_size is 0:
            print('processed %d of %d...' % (count, len(target_images)))
        count += 1
        if not reuse_images:
            input_images.remove(match_index)
    print('creating mosaic...')
    mosaic_image = createImageGrid(output_images, grid_size)
    return mosaic_image


def main():
    parser = argparse.ArgumentParser(description='Creates a photomosaic from input images')
    parser.add_argument('--target-image', dest='target_image', required=True)
    parser.add_argument('--input-folder', dest='input_folder', required=True)
    parser.add_argument('--grid-size', nargs=2, dest='grid_size', required=True)
    parser.add_argument('--output-file', dest='outfile', required=False)
    args = parser.parse_args()

    target_image = Image.open(args.target_image)
    print('reading input folder')
    input_images = getImages(args.input_folder)
    if input_images == []:
        print('No input images found in %s. Exiting.' % (args.input_folder,))
        exit()
    random.shuffle(input_images)

    grid_size = (int(args.grid_size[0]), int(args.grid_size[1]))
    output_filename = 'mosaic.png'
    if args.outfile:
        output_filename = args.outfile
    reuse_images = True
    resize_input = True

    print('starting photomosaic creation...')
    if not reuse_images:
        if grid_size[0]*grid_size[1] > len(input_images):
            print('grid size less than number of images')
            exit()
    if resize_input:
        print('resizing images...')
        dims = (int(target_image.size[0]/grid_size[0]), int(target_image[1]/grid_size[1]))
        print("max tile dims: %s" % (dims,))
        for img in input_images:
            img.thumbnail(dims)
        mosaic_image = createPhotomosaic(target_image, input_images, grid_size, reuse_images)
        mosaic_image.save(output_filename, 'PNG')
        print("saved output to %s" % (output_filename,))
        print('done.')


if __name__ == '__main__':
    main()