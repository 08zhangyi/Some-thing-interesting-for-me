import sys, random, argparse
from PIL import Image, ImageDraw


def createTiledImage(tile, dims):
    img = Image.new('RGB', dims)
    W, H = dims
    w, h = tile.size
    cols = int(W/w) + 1
    rows = int(H/h) + 1
    for i in range(rows):
        for j in range(cols):
            img.paste(tile, (j*w, i*h))
    return img


def createRandomTile(dims):
    img = Image.new('RGB', dims)
    draw = ImageDraw.Draw(img)
    r = int(min(*dims)/100)
    n = 1000
    for i in range(n):
        x, y = random.randint(0, dims[0]-r), random.randint(0, dims[1]-r)
        fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse((x-r, y-r, x+r, y+r), fill)
    return img


def createAutostereogram(dmap, tile):
    if dmap.mode is not 'L':
        dmap = dmap.convert('L')
    if not tile:
        tile = createRandomTile((100, 100))
    img = createTiledImage(tile, dmap.size)
    sImg = img.copy()
    pixD = dmap.load()
    pixS = sImg.load()
    cols, rows = sImg.size
    for j in range(rows):
        for i in range(cols):
            xshift = pixD[i, j]/10
            xpos = i - tile.size[0] + xshift
            if xpos > 0 and xpos < cols:
                pixS[i, j] = pixS[xpos, j]
    return sImg


def createSpacingDepthExample():
    tiles = [Image.open('test/a.png'), Image.open('test/b.png'), Image.open('test/c.png')]
    img = Image.new('RGB', (600, 400), (0, 0, 0))
    spacing = [10, 20, 40]
    for j, tile in enumerate(tiles):
        for i in range(8):
            img.paste(tile, (10+i*(100+10*j), 10+j*100))
    img.save('sdepth.png')


def createDepthMap(dims):
    dmap = Image.new('L', dims)
    dmap.paste(10, (200, 25, 300, 125))
    dmap.paste(30, (200, 150, 300, 250))
    dmap.paste(20, (200, 275, 300 ,375))
    return dmap


def createDepthShiftedImage(dmap, img):
    assert dmap.size == img.size
    sImg = img.copy()
    pixD = dmap.load()
    pixS = sImg.load()
    cols, rows = sImg.size
    for j in range(rows):
        for i in range(cols):
            xshift = pixD[i, j] / 10
            xpos = i - 140 + xshift
            if xpos > 0 and xpos < cols:
                pixS[i, j] = pixS[xpos, j]
    return sImg


def main():
    print('creating autostereogram...')
    parser = argparse.ArgumentParser(description="Autosterograms...")
    parser.add_argument('--depth', dest='dmFile', required=True)
    parser.add_argument('--tile', dest='tileFile', required=False)
    parser.add_argument('--out', dest='outFile', required=False)
    args = parser.parse_args()
    outFile = 'as.png'
    if args.outFile:
        outFile = args.outFile
    tileFile = False
    if args.tileFile:
        tileFile = Image.open(args.tileFile)
    dmImg = Image.open(args.dmFile)
    asImg = createAutostereogram(dmImg, tileFile)
    asImg.save(outFile)


if __name__ == '__main__':
    main()