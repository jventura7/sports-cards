import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math

def detectCardBorders(horizontal, vertical, img):
    # print out horizontal lines
    fig2, ax2 = plt.subplots()
    ax2.imshow(img, cmap=cm.gray)
    for line in horizontal:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * a)
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * a)
        ax2.plot((x1, x2), (y1, y2), 'red')

    rows, cols = img.shape
    ax2.axis((0, cols, rows, 0))
    ax2.set_title('Detected Lines')



    fig3, ax3 = plt.subplots()
    ax3.imshow(img, cmap=cm.gray)
    for line in vertical:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * a)
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * a)
        ax3.plot((x1, x2), (y1, y2), 'red')

    rows, cols = img.shape
    ax3.axis((0, cols, rows, 0))
    ax3.set_title('Detected Lines')

    middleOfImg = (rows // 2, cols // 2)

    # get top horizontal margins
    topHorizontalMargins = []
    currentPos = middleOfImg
    # while currentPos >= 0:
    return None, None

def detectVerticalLines(new_img,test_threshold):
    img = cv2.imread(new_img, 0)
    black_img = np.zeros(img.shape)
    median = np.median(img)
    edges = cv2.Canny(img, 2*median, 2.5*median)
    cv2.imshow('canny', edges)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold = test_threshold)
    N = lines.shape[0]
    vertical_lines = []
    row1, col1 = img.shape
    # fig1, ax1 = plt.subplots()
    # ax1.imshow(img, cmap=cm.gray)
    for i in range(N):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * a)
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * a)
        if x1 == x2 or abs(x1 - x2) <= 20:
            vertical_lines.append(lines[i])
    return vertical_lines, img, row1, col1

def detectHorizontalLines(new_img,test_threshold):
    img = cv2.imread(new_img, 0)
    black_img = np.zeros(img.shape)
    median = np.median(img)
    edges = cv2.Canny(img, 2*median, 2.5*median)
    cv2.imshow('canny', edges)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold = test_threshold)
    N = lines.shape[0]

    horizontal_lines = []
    row1, col1 = img.shape
    # fig1, ax1 = plt.subplots()
    # ax1.imshow(img, cmap=cm.gray)
    for i in range(N):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * a)
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * a)
        if y1 == y2 or abs(y1 - y2) <= 20:
            horizontal_lines.append(lines[i])
    return horizontal_lines, img, row1, col1
    
new_img = "psa9_183.jpg"
horizontalThreshold = 250
verticalThreshold = 400
horizontal_lines, img, row1, col1 = detectHorizontalLines(new_img, horizontalThreshold)
vertical_lines, img, row1, col1 = detectVerticalLines(new_img, verticalThreshold)

while len(horizontal_lines) < 3:
    horizontal_lines = horizontal_lines - 10
    horizontal_lines, img, row1, col1 = detectHorizontalLines(new_img, horizontalThreshold)
print(horizontalThreshold)

while len(vertical_lines) < 3:
    vertical_lines = vertical_lines - 10
    vertical_lines, img, row1, col1 = detectVerticalLines(new_img, verticalThreshold)
print(verticalThreshold)

vertical_margins, horizontal_margins = detectCardBorders(horizontal_lines, vertical_lines, img)

vertical_lines.sort(key=lambda x: x[0][0])
vertical_lines = vertical_lines[:2] + vertical_lines[-2:]

#print(vertical_lines)
length_bottom = vertical_lines[1][0][0] - vertical_lines[0][0][0]
length_top = vertical_lines[3][0][0] - vertical_lines[2][0][0]
print("Left:", length_bottom, "Right:", length_top)
print("Left to Right Ratio ", int(round((length_bottom / (length_bottom + length_top))*100)), ":", int(round((length_top / (length_top + length_bottom))*100)))

horizontal_lines.sort(key=lambda x: x[0][0])
horizontal_lines = horizontal_lines[:2] + horizontal_lines[-2:]
length_left = horizontal_lines[1][0][0] - horizontal_lines[0][0][0]
length_right = horizontal_lines[3][0][0] - horizontal_lines[2][0][0]
print("Bottom:", length_right, "Top: ", length_left)
print("Bottom to Top Ratio ", int(round((length_left / (length_left + length_right))*100)), ":", int(round((length_right / (length_left + length_right))*100)))


borders = horizontal_lines + vertical_lines
fig2, ax2 = plt.subplots()
ax2.imshow(img, cmap=cm.gray)
for line in borders:
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 5000 * (-b))
    y1 = int(y0 + 5000 * a)
    x2 = int(x0 - 5000 * (-b))
    y2 = int(y0 - 5000 * a)
    for border in borders:
        ax2.plot((x1, x2), (y1, y2), 'red', linewidth=0)
ax2.axis((0, col1, row1, 0))
ax2.set_title('Corner Detection')
ax2.set_axis_off()

#fig2.savefig('houghlines.png')

left_x = (vertical_lines[0][0][0] + vertical_lines[1][0][0]) // 2
right_x = (vertical_lines[2][0][0] + vertical_lines[3][0][0]) // 2

top_y = (horizontal_lines[0][0][0] + horizontal_lines[1][0][0]) // 2
bottom_y = (horizontal_lines[2][0][0] + horizontal_lines[3][0][0]) // 2

'''
print("Vertical Line 1 (x-coordinate): " + str(vertical_lines[0][0][0]))
print("Vertical Line 2 (x-coordinate): " + str(vertical_lines[1][0][0]))
print("Vertical Line 3 (x-coordinate): " + str(vertical_lines[2][0][0]))
print("Vertical Line 4 (x-coordinate): " + str(vertical_lines[3][0][0]))
print() 
print("Horizontal Line 1 (y-coordinate): " + str(horizontal_lines[0][0][0]))
print("Horizontal Line 2 (y-coordinate): " + str(horizontal_lines[1][0][0]))
print("Horizontal Line 3 (y-coordinate): " + str(horizontal_lines[2][0][0]))
print("Horizontal Line 4 (y-coordinate): " + str(horizontal_lines[3][0][0]))
print()
print("Top left corner: (" + str(vertical_lines[0][0][0]) + ", " + str(horizontal_lines[0][0][0]) + ")")
print("Bottom left corner: (" + str(vertical_lines[0][0][0]) + ", " + str(horizontal_lines[3][0][0]) + ")")
print("Top right corner: (" + str(vertical_lines[3][0][0]) + ", " + str(horizontal_lines[0][0][0]) + ")")
print("Bottom right corner: (" + str(vertical_lines[3][0][0]) + ", " + str(horizontal_lines[3][0][0]) + ")")
print()
print("Top left center point: (" + str(left_x) + ", " + str(top_y) + ")")
print("Bottom left center point: (" + str(left_x) + ", " + str(bottom_y) + ")")
print("Top right center point: (" + str(right_x) + ", " + str(top_y) + ")")
print("Bottom right center point: (" + str(right_x) + ", " + str(bottom_y) + ")")
'''

ax2.add_patch(patches.Rectangle((left_x, top_y), vertical_lines[0][0][0] - left_x, horizontal_lines[0][0][0] - top_y, linewidth=1, edgecolor='r', facecolor='none'))
ax2.add_patch(patches.Rectangle((left_x, bottom_y), vertical_lines[0][0][0] - left_x, horizontal_lines[3][0][0] - bottom_y, linewidth=1, edgecolor='r', facecolor='none'))
ax2.add_patch(patches.Rectangle((right_x, top_y), vertical_lines[3][0][0] - right_x, horizontal_lines[0][0][0] - top_y, linewidth=1, edgecolor='r', facecolor='none'))
ax2.add_patch(patches.Rectangle((right_x, bottom_y), vertical_lines[3][0][0] - right_x, horizontal_lines[3][0][0] - bottom_y, linewidth=1, edgecolor='r', facecolor='none'))
plt.show()

corner1 = img[int(horizontal_lines[0][0][0]):int(top_y) , int(vertical_lines[0][0][0]):int(left_x)]
corner2 = img[int(bottom_y):int(horizontal_lines[3][0][0]) , int(vertical_lines[0][0][0]):int(left_x)]
corner3 = img[int(horizontal_lines[0][0][0]):int(top_y) , int(right_x):int(vertical_lines[3][0][0])]
corner4 = img[int(bottom_y):int(horizontal_lines[3][0][0]) , int(right_x):int(vertical_lines[3][0][0])]

ret1, otsu1 = cv2.threshold(corner1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2, otsu2 = cv2.threshold(corner2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3, otsu3 = cv2.threshold(corner3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret4, otsu4 = cv2.threshold(corner4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

horiStack1 = np.hstack((corner1, corner3))
horiStack2 = np.hstack((corner2, corner4))
stack = np.vstack((horiStack1, horiStack2))

horiOtsuStack1 = np.hstack((otsu1, otsu3))
horiOtsuStack2 = np.hstack((otsu2, otsu4))
stackOtsu = np.vstack((horiOtsuStack1, horiOtsuStack2))

cornerPixels1 = []
cornerPixels2 = []
cornerPixels3 = []
cornerPixels4 = []

#multiply the otsu threshold by 0.8


for i in range(len(otsu1)):
    for j in range(len(otsu1[i])):
        if otsu1[i][j] == 255:
            cornerPixels1.append(corner1[i][j])

for i in range(len(otsu2)):
    for j in range(len(otsu2[i])):
        if otsu2[i][j] == 255:
            cornerPixels2.append(corner2[i][j])

for i in range(len(otsu3)):
    for j in range(len(otsu3[i])):
        if otsu3[i][j] == 255:
            cornerPixels3.append(corner3[i][j])

for i in range(len(otsu4)):
    for j in range(len(otsu4[i])):
        if otsu4[i][j] == 255:
            cornerPixels4.append(corner4[i][j])

Pixels1 = np.array(cornerPixels1)
Pixels2 = np.array(cornerPixels2)
Pixels3 = np.array(cornerPixels3)
Pixels4 = np.array(cornerPixels4)

print("CORNERS DETECTED USING HOUGH TRANSFORM TECHNIQUE")
print("Top Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(corner1))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner1))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner1))))
print("Bottom Left Corner:\t\t" , "Average: ", str("{:.2f}".format(np.mean(corner2))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner2))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner2))))
print("Top Right Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(corner3))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner3))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner3))))
print("Bottom Right Corner:\t", "Average: ", str("{:.2f}".format(np.mean(corner4))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner4))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner4))))

print("\nBELOW ARE CORNERS USING OTSU METHOD")
print("Top Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels1))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels1))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels1))))
print("Bottom Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels2))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels2))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels2))))
print("Top Right Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels3))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels3))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels3))))
print("Bottom Right Corner:\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels4))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels4))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels4))))

'''
lines = cv2.HoughLines(corner1, 1, math.pi/2, 2, None, 30, 1);
for line in lines:
    pt1 = (line[0],line[1])
    pt2 = (line[2],line[3])
    cv2.line(img, pt1, pt2, (0,0,255), 3)
'''
'''
print(type(img))
print(img)
cornerPixel1 = np.array(cornerPixels1)
print(type(Pixels1))
print(Pixels1)
lines = cv2.HoughLinesP(otsu1, 1, np.pi/180, 30, maxLineGap=250)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(otsu1, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("corner1", corner1)
cv2.imshow("corner2", corner2)
cv2.imshow("corner3", corner3)
cv2.imshow("corner4", corner4)
cv2.imshow("otsu corner1: " , otsu1)
cv2.imshow("otsu corner2 otsu: " , otsu2)
cv2.imshow("otsu corner3 otsu: " , otsu3)
cv2.imshow("otsu corner4 otsu: " , otsu4)
cv2.imshow("pixels1", otsu1)
cv2.imshow("Corners", stack)
cv2.imshow("Otsu Corners", stackOtsu)
'''
cv2.waitKey(0)