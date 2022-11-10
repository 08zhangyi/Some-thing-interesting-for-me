import os
from pathlib import Path
import cv2 as cv

PAD = 5


def find_transient(image, diff_image, pad):
    transient = False
    height, width = diff_image.shape
    cv.rectangle(image, (PAD, PAD), (width-PAD, height-PAD), 255, 1)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(diff_image)
    if pad < maxLoc[0] < width-pad and pad < maxLoc[1] < height-pad:
        cv.circle(image, maxLoc, 10, 255, 0)
        transient = True
    return transient, maxLoc


def main():
    night1_files = sorted(os.listdir('night_1_registered_transients'))
    night2_files = sorted(os.listdir('night_2'))
    path1 = Path.cwd() / 'night_1_registered_transients'
    path2 = Path.cwd() / 'night_2'
    path3 = Path.cwd() / 'night_1_2_transients'

    for i, _ in enumerate(night1_files[:-1]):
        img1 = cv.imread(str(path1 / night1_files[i]), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(str(path2 / night2_files[i]), cv.IMREAD_GRAYSCALE)

        diff_imgs1_2 = cv.absdiff(img1, img2)
        cv.imshow('Difference', diff_imgs1_2)
        cv.waitKey(2000)

        temp = diff_imgs1_2.copy()
        transient1, transient_loc1 = find_transient(img1, temp, PAD)
        cv.circle(temp, transient_loc1, 10, 0, -1)

        transient2, _ = find_transient(img1, temp, PAD)

        if transient1 or transient2:
            print('\nTRANSIENT DETECTED between {} and {}\n'.format(night1_files[i], night2_files[i]))
            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(img1, night1_files[i], (10, 25), font, 1, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(img1, night2_files[i], (10, 55), font, 1, (255, 255, 255), 1, cv.LINE_AA)
            blended = cv.addWeighted(img1, 1, diff_imgs1_2, 1, 0)
            cv.imshow('Surveyed', blended)
            cv.waitKey(2500)

            out_filename = '{}_DETECTED.png'.format(night1_files[i][:-4])
            cv.imwrite(str(path3/ out_filename), blended)
        else:
            print('\nNo transient detected between {} and {}\n'.format(night1_files[i], night2_files[i]))


if __name__ == '__main__':
    main()