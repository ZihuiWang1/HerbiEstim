import csv
import os
import cv2
import imutils
import numpy as np
import math
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="Split multiple-leaf images to single-leaf images and standardize to 256*256",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input",  type= Path, required=True, help="Path to the input directory")
parser.add_argument("-o", "--output", type= Path, default='./imgs_standardized', help="Path to the output directory")

args = parser.parse_args()

def angle(img):
    # convert to grayscale
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #inverted binary threshold
    _, thresh = cv2.threshold(img_gs, 250, 1, cv2.THRESH_BINARY_INV)
    #From a matrix of pixels to a matrix of coordinates of non-black points.
    #(note: mind the col/row order, pixels are accessed as [row, col]
    #but when we draw, it's (x, y), so have to swap here or there)
    mat = np.argwhere(thresh != 0)
    #swap here
    mat[:, [0, 1]] = mat[:, [1, 0]]
    # convert type for PCA
    mat = np.array(mat).astype(np.float32)
    #mean (e. g. the geometrical center) and eigenvectors (e. g. directions of principal components)
    m, e = cv2.PCACompute(mat, mean = np.array([]))

    #scale our primary axis by 100,
    center = tuple(m[0])
    endpoint1 = tuple(m[0] + e[0]*100)

    ## calculate the angle in degree to horizontal
    y = center[1] - endpoint1[1]
    x = endpoint1[0] - center[0]

    if x < 0 and y < 0:
        radianA = math.atan2(abs(y), abs(x))
    elif x > 0 and y < 0:
        radianA = math.atan2((-y), (-x))
    else:
        radianA = math.atan2(y,x)

    angleHor = np.rad2deg(radianA)
    return(angleHor)

def getmask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return(threshInv)


if not os.path.exists(args.output):
    os.makedirs(args.output)
dirname = os.path.join(args.output,'test')
if not os.path.exists(dirname):
    os.makedirs(dirname)


# remove .DS_Store file in the folder
img = os.listdir(args.input)
if '.DS_Store' in img:
    img.remove('.DS_Store')

with open(os.path.join(args.output, 'resize_ratio.csv'), "w") as resizcsv:
    wresizcsv = csv.writer(resizcsv)
    wresizcsv.writerow(['img_tag', 'img_num', 'resize_ratio'])

    for imagepath in img:
        pathfrom = os.path.join(args.input, imagepath)
        image = cv2.imread(pathfrom)

        #enlarge a bit to draw contours easier
        image = cv2.copyMakeBorder(image, top=30, bottom=30, left=30, right=30,
                                       borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply a "small" blur to the image, to remove those salt-and-pepper style noise
        blurred = cv2.medianBlur(gray, 15)

        # apply Otsu's automatic thresholding to remove the impure background
        (T, threshInv) = cv2.threshold(blurred, 0, 255,
                                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        f = cv2.bitwise_and(image, image, mask=threshInv)
        b = np.ones_like(image, np.uint8) * 255
        cv2.bitwise_not(b, b, mask=threshInv)
        image_th = f + b

        # apply Canny edge detection using automatically determined threshold
        auto = imutils.auto_canny(threshInv)

        # apply dilate
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(auto, kernel, iterations=1)

        # find contours in the image, but this time keep only the EXTERNAL contours
        cntss = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cntss)
        cnts = cntss[0] if len(cntss) == 2 else cntss[1]

        # loop to extract each contours to a new square image
        image_number = 0
        for c in cnts:
            # draw on a blank image
            canvas = np.zeros(image_th.shape[0:2], dtype=np.uint8)
            cv2.drawContours(canvas, [c], -1, (255, 255, 255), -1, cv2.LINE_AA)
            res = cv2.bitwise_and(image_th, image_th, mask=canvas)

            # crate the white background of the same size of original image
            wbg = np.ones_like(image_th, np.uint8) * 255
            cv2.bitwise_not(wbg, wbg, mask=canvas)

            # overlap the resulted cropped image on the white background
            dst = wbg + res

            # extract ROI and make object in the CENTER of the new image
            x, y, w, heig = cv2.boundingRect(c)
            cx = x + round(w / 2)
            cy = y + round(heig / 2)

            if w >= heig:
                heig = round(round(w / 2) * 1.5)
            else:
                heig = round(round(heig / 2) * 1.5)

            y1 = cy - heig
            x1 = cx - heig
            y2 = cy + heig
            x2 = cx + heig

            if y1 < 0:
                dst = cv2.copyMakeBorder(dst, top=(-y1), bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])
                y1 = 0
                y2 = heig + heig

            if x1 < 0:
                dst = cv2.copyMakeBorder(dst, top=0, bottom=0, left=(-x1), right=0, borderType=cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])
                x1 = 0
                x2 = heig + heig

            if y2 > (image.shape[0]):
                dst = cv2.copyMakeBorder(dst, top=0, bottom=(y2 - image.shape[0]), left=0, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

            if x2 > (image.shape[1]):
                dst = cv2.copyMakeBorder(dst, top=0, bottom=0, left=0, right=(x2 - image.shape[1]),
                                             borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

            ROI = dst[y1:y2, x1:x2]

            # only keep objects with height larger than 128 pixes, so that noises won't be a separate img
            if heig > 128:

                # rotate the object to vertical i.e. make every leaf vertical
                ang = 90 - angle(ROI)
                M = cv2.getRotationMatrix2D((ROI.shape[0]/2, ROI.shape[1]/2), ang, 1.0)

                mask = getmask(ROI)
                change = cv2.warpAffine(ROI, M, (ROI.shape[0], ROI.shape[1]))
                changmask = cv2.warpAffine(mask, M, (ROI.shape[0], ROI.shape[1]))
                image_synthetic = cv2.bitwise_and(change, change, mask=changmask)

                b = np.ones_like(ROI, np.uint8) * 255
                cv2.bitwise_not(b, b, mask=changmask)
                ROI = image_synthetic + b

                # resized the ROI to 256*256, and record the size change
                length, height, _ = ROI.shape
                ROIsized = cv2.resize(ROI, (256, 256), interpolation=cv2.INTER_AREA)

                dd1 = height / 256

                wresizcsv.writerow([imagepath, image_number, dd1])

                # write the img
                nam = "{}_{}.png".format(imagepath, image_number)
                pathdir = os.path.join(dirname, nam)
                cv2.imwrite(pathdir, ROIsized)
                image_number += 1
                print(nam)


