import cv2
import numpy as np

'''
takes initial baseball card and crops out the background using canny and contour
'''

# Load image, convert to grayscale, and find edges
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

# Find contour and sort by contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = image[y:y+h, x:x+w]
    break

'''
runs canny on the cropped image and then contour again to get the inner object
'''
canny = cv2.Canny(ROI, 15, 10, 1)

# Find contours in the image
cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

contours = []

threshold_min_area = 400
for c in cnts:
    area = cv2.contourArea(c)
    if area > threshold_min_area:
        cv2.drawContours(ROI,[c], 0, (0,255,0), 3)
        contours.append(c)
'''
# Find the edges in the image using canny detector
edges = cv2.Canny(ROI,200, 250)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180,15, minLineLength=50, maxLineGap=20)

# Draw lines on the image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(ROI,(x1,y1),(x2,y2),(0,255,0),2)
    #x1, y1, x2, y2 = line[0]
    #cv2.line(ROI, (x1, y1), (x2, y2), (255, 0, 0), 3)
'''
cv2.imshow('canny',canny)
cv2.imshow('output',ROI)
cv2.imwrite('output.jpg',ROI)
cv2.waitKey()
