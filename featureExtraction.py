# author: Noah Sanzone
# ECE 4805
# 02/23/2022
# This will extract the feature vector of each card and place the vector into
    # a CSV file where each row is a feature vector.

import sys
import cv2
import csv
import os
from os.path import exists

import hough_transform
from hough_transform import *


def find_grade(filename):
    split = filename.split("_", 1)[0]
    return split[3:]


fileName = input('What will be the name of the .csv file?: ')

# first, need to see if this file already exists
if os.path.exists(fileName):
    decision = input('This file already exists. Do you want to overwrite? [Y/N]: ')
    if decision == 'Y':# if it exists, overwrite the file by clearing the entire contents
        fileVar = open(fileName, 'r+')
        fileVar.truncate(0)
        fileVar.close()
    elif decision == 'N':
        fileName = input('Since you do not want to overwrite, what will be the name of the .csv file?: ')

# Before creating the user-specified file, need to determine which folder contains the images
picDir = input('Which directory contains the images?: ')
#print('picDir:', picDir)

# need to make sure the folder of pictures exists
while not os.path.isdir(picDir):
    picDir = input('This directory does not exist. Try another directory: ')

folder = os.listdir(picDir)
num = len(folder)
# print('num:', num)

# now that we have verified the name of the csv file and the existence of picDir
    # we can begin extracting features from each image

# 1. Create the .csv file with the user-specified name
#f = open(fileName, 'w', newline='')
with open(fileName, 'w', newline='') as f:

    # 2. Setup/initialize the .csv file with proper headings
    writer = csv.writer(f) # create the csv writer

    headerRow = ['fileName', 'grade', 'Vertical MR', 'Horizontal MR', 'TL STD', 'TR STD',
                 'BL STD', 'BR STD', 'LS MC', 'BS MC']

    writer.writerow(headerRow)

    # loop through each image in the user specified directory
    for file in os.listdir(picDir):
        filePath = os.path.join(picDir, file)

        if os.path.isfile(filePath):  # checking to make sure it is a file
            img = cv2.imread(filePath, 0)


            name = file
            print("name:", name)
            grade = find_grade(name)

            verticalRatio, horizontalRatio, c1, c2, c3, c4 = hough_transform.test(img)

            row = [name, grade, verticalRatio, horizontalRatio, c1, c2, c3, c4]
            writer.writerow(row)


            # cv2.imshow(file, img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






# need to go through all files in all folders of "Images"

# for each folder within "Images"

    # for each images within folder
        # Vertical margin ratio
        # Horizontal margin ratio
        # Standard Deviation of each corner intensity (all 4 corners individually)
        # Count of black pixels from Otsu method (all 4 corners individually)
        # Left side change in margin size
        # Bottom side change in margin size

