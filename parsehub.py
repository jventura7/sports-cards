#author: Noah Sanzone
#ECE 4805
#This will read the csv file from parsehub .csv file and save the images in a folder
    #Eliminates the need for tabsave chrome extension

import sys
import os
import urllib.request
from os.path import exists
import csv

fileName = sys.argv[1]

if exists(fileName):
    print("fileName exists")
    print(fileName)
    grade = fileName.split("grade", 1)[1].split(".", 1)[0]
    print(grade)

    if (os.path.isdir('Images/collectors' + grade)):
        print("collectors" + grade + " directory exists")

        #open .csv file
        with open(fileName, 'r') as csv_file:
            datareader = csv.reader(csv_file)
            i = 0
            for row in datareader:
                if i != 0 and row[1] != "" and row[3] != "":
                    #print("i: ", i)
                    #print("card_name: ", row[1])
                    #print("card_image: ", row[3])
                    name = row[1]
                    url = row[3]
                    urllib.request.urlretrieve(url, "Images/" + "collectors" + grade + "/psa" + grade + "_" +str(i) + ".jpg")
                i+=1
    else:
        print("collectors" + grade + "directory does not exist")
else:
    print("fileName does not exist")



#access specific column in .csv file