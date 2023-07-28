import cv2
import numpy as np
import random as rd
import os
import math
import csv
import argparse
from pathlib import Path

def getmask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return(threshInv)

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

parser = argparse.ArgumentParser(description="Create artificial herbivore damages on intact leaves",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input",  type= Path, required=True, help="Path to the directory of intact leaves i.e. input")
parser.add_argument("-o", "--output", type= Path, default='./imgs_synthesis', help="Path to the output directory")
parser.add_argument("-n", "--number", type=int, default=5000, help="The number of artificial leaf images ")

args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)
dir_to = os.path.join(args.output,'train')
if not os.path.exists(dir_to):
    os.makedirs(dir_to)

f=open(os.path.join(args.output,'stats.csv'),'w')
w=csv.writer(f)
w.writerow(['syn_img', 'leaf_area_intact', 'leaf_area_damage','artificial_damage_pct'])

list_h = list(listdir_nohidden(args.input))
image_number = 0
for num in range(args.number):
    # select an intact image randomly
    indice = rd.randint(1, len(list_h) - 1)
    path_h = os.path.join(args.input,list_h[indice])
    image = cv2.imread(path_h)
    imageresized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    mask = getmask(imageresized)
    d1 = np.count_nonzero(mask)
    pixels = np.transpose(np.nonzero(mask))

    #create synthetic damage on the intact leaf
    img_syn = image.copy()

    # by draw circles, polygon or both
    # randomly choose 1(drawing polygons),2(drawing circles) or 3(drawing both)
    c = rd.randint(1, 3)
    if c == 1 or c == 3:
        # drawing 1-6 polygons
        n = rd.randint(1, 6)
        for i in range(n):
            pix = rd.choice(pixels)
            xcent = pix[1]
            ycent = pix[0]

            ang = 0
            raio = rd.randint(5, 20)

            points = []
            while ang < 360:
                x = xcent + int(raio * math.sin(math.radians(ang)))
                y = ycent + int(raio * math.cos(math.radians(ang)))
                points.append([y, x])

                raio += rd.randint(-4, 5)
                ang += rd.randint(10, 40)

            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img_syn, [points], isClosed=True, color=(255, 255, 255), thickness=1)
            # fill it
            cv2.fillPoly(img_syn, [points], color=(255, 255, 255))

    if c == 2 or c == 3:
        # drawing 5-20 small circles
        n = rd.randint(5, 20)
        for i in range(n):
            pix = rd.choice(pixels)
            centerpix = (pix[1], pix[0])

            r = rd.randint(1, 5)
            cv2.circle(img_syn, centerpix, r, (255, 255, 255), -1)

    mask2 = getmask(img_syn)
    d2 = np.count_nonzero(mask2)

    image_synthetic = cv2.bitwise_and(img_syn, img_syn, mask=mask2)
    b = np.ones_like(img_syn, np.uint8) * 255
    cv2.bitwise_not(b, b, mask=mask2)
    image_synthetic1 = image_synthetic + b

    # combine img and syn img as training data
    imgfina = np.concatenate([image, image_synthetic1], 1)
    nam = "syn_{}.png".format(image_number)
    fina2 = os.path.join(dir_to, nam)

    cv2.imwrite(fina2, imgfina)

    d = (d1 - d2) / d1
    w.writerow([image_number, d1, d2, d])

    print(image_number)
    image_number += 1


