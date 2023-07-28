import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import csv

parser = argparse.ArgumentParser(description="Calculate leaf damage based on reconstructed leaves ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-s", "--standardized",  type= Path, default='./imgs_standardized', help="Path to the standardized images")
parser.add_argument("-p", "--imgspredict", type= Path, default='./imgs_predicted', help="Path of predicted images")
parser.add_argument("-n", '--name', type=str, default='experiment_name', help='model name')
parser.add_argument("-d", '--dpi', type= int, default=300, help="The resolution of images in dpi, default is 300 dpi")
parser.add_argument('--notscanned', action='store_false', help="Include the argument if images are not scanned")

args = parser.parse_args()

def getmask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return (threshInv)

dpivalue = (args.dpi / 2.54) ** 2  # dpi

with open("result.csv", "w") as predcsv:
    wpredcsv = csv.writer(predcsv)
    if args.notscanned:
        wpredcsv.writerow(['img.tag', 'ind.leaf', 'LA(cm2)', 'intact.LA(cm2)', 'damage(%)'])
    else:
        wpredcsv.writerow(['img.tag', 'ind.leaf', 'num_pixel', 'intact.num_pixel', 'damage(%)'])

    with open(os.path.join(args.standardized, "resize_ratio.csv"), mode='r', encoding='UTF-8-sig') as f_input:
        csv_input = csv.reader(f_input)
        next(csv_input)
        for row in csv_input:
            fake = os.path.join(args.imgspredict, args.name, 'test_latest', 'images', str(row[0])+'_'+str(row[1])+'_fake.png' )
            real = os.path.join(args.imgspredict, args.name, 'test_latest', 'images', str(row[0])+'_'+str(row[1])+'_real.png' )
            imagereal = cv2.imread(real)
            mask = getmask(imagereal)
            d1 = np.count_nonzero(mask)

            imagefake = cv2.imread(fake)
            mask2 = getmask(imagefake)
            d2 = np.count_nonzero(mask2)

            if args.notscanned:
                act_area = round(d1 * float(row[2]) * float(row[2]) / dpivalue, 3)
                intact_area = round(d2 * float(row[2]) * float(row[2]) / dpivalue, 3)
            else:
                act_area = int(round(d1 * float(row[2]) * float(row[2]), 0))
                intact_area = int(round(d2 * float(row[2]) * float(row[2]), 0))

            proportation = round((intact_area - act_area) / intact_area, 3)
            wpredcsv.writerow([row[0], row[1], act_area, intact_area, proportation])