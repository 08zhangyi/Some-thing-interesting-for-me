import zipfile, os


def backupToZip(folder):
    folder = os.path.abspath(folder)
    number = 1
    while True:
        zipFilename = os.path.basename(folder) + '_' + str(number) + '.zip'
        if not os.path.exists(zipFilename):
            break
        number = number + 1

    print(f'Creating {zipFilename}...')
    backupZip = zipfile.ZipFile(zipFilename, 'w')
    for foldername, subfolders, filenames in os.walk(folder):
        print(f'Adding files in {foldername}...')
        backupZip.write(foldername)
        for filename in filenames:
            newBase = os.path.basename(folder) + '_'
            if filename.startwith(newBase) and filename.endswith('.zip'):
                continue
            backupZip.write(os.path.join(foldername, filename))
    print('Done')


backupToZip('C:\\delicious')