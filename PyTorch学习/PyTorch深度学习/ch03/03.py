import glob
import os
import numpy as np

path = ''
files = glob(os.path.join(path, '*/*.jpg'))
print('Total no of images {len(files)}')
no_of_images = len(files)
shuffle = np.random.permutation(no_of_images)
os.mkdir(os.path.join(path, 'valid'))

for t in ['train', 'valid']:
    for folder in ['dog/', 'cat/']:
        os.mkdir(os.path.join(path, t, folder))

for i in shuffle[:2000]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(path, 'valid', folder, image))

for i in shuffle[2000:]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(path, 'train', folder, image))