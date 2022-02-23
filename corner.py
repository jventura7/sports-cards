import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math


pull_img = "image1.png"
new_img = "resized_image1.png"

image = Image.open(pull_img)
resized_img = image.resize((2000, 1000))
resized_img.save(new_img)

img = cv2.imread(new_img, 0)
black_img = np.zeros(img.shape)
median = np.median(img)
edges = cv2.Canny(img, 2*median, 2.5*median)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=300) 
N = lines.shape[0]

horizontal_lines = []
vertical_lines = []
row1, col1 = img.shape
fig1, ax1 = plt.subplots()
ax1.imshow(img, cmap=cm.gray)
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
    if x1 == x2 or abs(x1 - x2) <= 20:
        vertical_lines.append(lines[i])
    ax1.plot((x1, x2), (y1, y2), 'red')
row1, col1 = img.shape
ax1.axis((0, col1, row1, 0))
ax1.set_title('Detected Lines')
ax1.set_axis_off()

vertical_lines.sort(key=lambda x: x[0][0])
vertical_lines = vertical_lines[:2] + vertical_lines[-2:]

#print(vertical_lines)
length_bottom = vertical_lines[1][0][0] - vertical_lines[0][0][0]
length_top = vertical_lines[3][0][0] - vertical_lines[2][0][0]
#print("Left:", length_bottom, "Right:", length_top)
#print("Left to Right Ratio ", int(round((length_bottom / (length_bottom + length_top))*100)), ":", int(round((length_top / (length_top + length_bottom))*100)))

horizontal_lines.sort(key=lambda x: x[0][0])
horizontal_lines = horizontal_lines[:2] + horizontal_lines[-2:]
#print(horizontal_lines)
length_left = horizontal_lines[1][0][0] - horizontal_lines[0][0][0]
length_right = horizontal_lines[3][0][0] - horizontal_lines[2][0][0]
#print("Bottom:", length_right, "Top: ", length_left)
#print("Bottom to Top Ratio ", int(round((length_left / (length_left + length_right))*100)), ":", int(round((length_right / (length_left + length_right))*100)))

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
#print("Min: " + str(np.min(corner1)))
#print("Max: " + str(np.max(corner1)))
#print("Average: " + str(np.mean(corner1)))
#print("Standard Deviation: " + str(np.std(corner1)))
#print("Variance: " + str(np.var(corner1)))

#print()
print("Bottom Left Corner:\t" , "Average: ", str("{:.2f}".format(np.mean(corner2))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner2))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner2))))
#print("Min: " + str(np.min(corner2)))
#print("Max: " + str(np.max(corner2)))
#print("Average: " + str(np.mean(corner2)))
#print("Standard Deviation: " + str(np.std(corner2)))
#print("Variance: " + str(np.var(corner2)))

#print()
print("Top Right Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(corner3))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner3))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner3))))
#print("Min: " + str(np.min(corner3)))
#print("Max: " + str(np.max(corner3)))
#print("Average: " + str(np.mean(corner3)))
#print("Standard Deviation: " + str(np.std(corner3)))
#print("Variance: " + str(np.var(corner3)))

#print()
print("Bottom Right Corner:\t", "Average: ", str("{:.2f}".format(np.mean(corner4))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(corner4))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner4))))
#print("Min: " + str(np.min(corner4)))
#print("Max: " + str(np.max(corner4)))
#print("Average: " + str(np.mean(corner4)))
#print("Standard Deviation: " + str(np.std(corner4)))
#print("Variance: " + str(np.var(corner4)))     

print("\nBELOW ARE CORNERS USING OTSU METHOD")
print("Top Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels1))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels1))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels1))))
#print("Min: " + str(np.min(cornerPixels1)))
#print("Max: " + str(np.max(cornerPixels1)))
#print("Average: " + str(np.mean(cornerPixels1)))
#print("Standard Deviation: " + str(np.std(cornerPixels1)))
#print("Variance: " + str(np.var(cornerPixels1)))

print("Bottom Left Corner:\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels2))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels2))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels2))))
#print("Min: " + str(np.min(cornerPixels2)))
#print("Max: " + str(np.max(cornerPixels2)))
#print("Average: " + str(np.mean(cornerPixels2)))
#print("Standard Deviation: " + str(np.std(cornerPixels2)))
#print("Variance: " + str(np.var(cornerPixels2)))

print("Top Right Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels3))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels3))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels3))))
#print("Min: " + str(np.min(cornerPixels3)))
#print("Max: " + str(np.max(cornerPixels3)))
#print("Average: " + str(np.mean(cornerPixels3)))
#print("Standard Deviation: " + str(np.std(cornerPixels3)))
#print("Variance: " + str(np.var(cornerPixels3)))

print("Bottom Right Corner:\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels4))), "\t", "Standard Deviation: ",  str("{:.2f}".format(np.std(cornerPixels4))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels4))))
#print("Min: " + str(np.min(cornerPixels4)))
#print("Max: " + str(np.max(cornerPixels4)))
#print("Average: " + str(np.mean(cornerPixels4)))
#print("Standard Deviation: " + str(np.std(cornerPixels4)))
#print("Variance: " + str(np.var(cornerPixels4)))
'''
lines = cv2.HoughLines(corner1, 1, math.pi/2, 2, None, 30, 1);
for line in lines:
    pt1 = (line[0],line[1])
    pt2 = (line[2],line[3])
    cv2.line(img, pt1, pt2, (0,0,255), 3)
'''

lines = cv2.HoughLines(otsu1, rho=1, theta=np.pi/180, threshold=10) 

horizontal_lines1 = []
vertical_lines1 = []
for i in range(lines.shape[0]):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b)) - 2
    y1 = int(y0 + 1000 * a) - 2
    x2 = int(x0 - 1000 * (-b)) - 2
    y2 = int(y0 - 1000 * a) - 2
    #print(theta)
    if theta <= 0.1:
        vertical_lines1.append([x1, y1, x2, y2, rho])
    if theta >= 1.5 and theta <= 1.72:
        horizontal_lines1.append([x1, y1, x2, y2, rho])
        
horizontal_lines1.sort(key=lambda x: x[-1])
horizontal_lines1 = horizontal_lines1[:1]
vertical_lines1.sort(key=lambda x: x[-1])
vertical_lines1 = vertical_lines1[:1]

#print(horizontal_lines1[0])
#print(vertical_lines1[0])

cv2.line(corner1, (horizontal_lines1[0][0], horizontal_lines1[0][1]), (horizontal_lines1[0][2], horizontal_lines1[0][3]), color=(0,255,0), thickness=1)
cv2.line(corner1, (vertical_lines1[0][0], vertical_lines1[0][1]), (vertical_lines1[0][2], vertical_lines1[0][3]), color=(0,255,0), thickness=1)

cv2.line(otsu1, (horizontal_lines1[0][0], horizontal_lines1[0][1]), (horizontal_lines1[0][2], horizontal_lines1[0][3]), (255,0,0), 1)
cv2.line(otsu1, (vertical_lines1[0][0], vertical_lines1[0][1]), (vertical_lines1[0][2], vertical_lines1[0][3]), (255,0,0), 1)

lines = cv2.HoughLines(otsu2, rho=1, theta=np.pi/180, threshold=10) 

horizontal_lines2 = []
vertical_lines2 = []
for i in range(lines.shape[0]):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 5000 * (-b)) - 2
    y1 = int(y0 + 5000 * a) + 2
    x2 = int(x0 - 5000 * (-b)) - 2
    y2 = int(y0 - 5000 * a) + 2
    #print(theta)
    if theta <= 0.1:
        vertical_lines2.append([x1, y1, x2, y2, rho])
    if theta >= 1.5 and theta <= 1.72:
        horizontal_lines2.append([x1, y1, x2, y2, rho])
        
horizontal_lines2.sort(key=lambda x: x[-1])
horizontal_lines2 = horizontal_lines2[-1:]
vertical_lines2.sort(key=lambda x: x[-1])
vertical_lines2 = vertical_lines2[:1]
#print("horizontal : ", horizontal_lines2[0])
#print("vertical : ", vertical_lines2[0])

cv2.line(corner2, (horizontal_lines2[0][0], horizontal_lines2[0][1]), (horizontal_lines2[0][2], horizontal_lines2[0][3]), color=(0,255,0), thickness=1)
cv2.line(corner2, (vertical_lines2[0][0], vertical_lines2[0][1]), (vertical_lines2[0][2], vertical_lines2[0][3]), color=(0,255,0), thickness=1)

cv2.line(otsu2, (horizontal_lines2[0][0], horizontal_lines2[0][1]), (horizontal_lines2[0][2], horizontal_lines2[0][3]), (255,0,0), 1)
cv2.line(otsu2, (vertical_lines2[0][0], vertical_lines2[0][1]), (vertical_lines2[0][2], vertical_lines2[0][3]), (255,0,0), 1)

lines = cv2.HoughLines(otsu3, rho=1, theta=np.pi/180, threshold=10) 

horizontal_lines3 = []
vertical_lines3 = []
for i in range(lines.shape[0]):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 5000 * (-b)) + 2
    y1 = int(y0 + 5000 * a) - 2
    x2 = int(x0 - 5000 * (-b)) + 2
    y2 = int(y0 - 5000 * a) - 2
    #print(theta)
    if theta <= 0.1:
        vertical_lines3.append([x1, y1, x2, y2, rho])
    if theta >= 1.5 and theta <= 1.72:
        horizontal_lines3.append([x1, y1, x2, y2, rho])
        
horizontal_lines3.sort(key=lambda x: x[-1])
horizontal_lines3 = horizontal_lines3[:1]
vertical_lines3.sort(key=lambda x: x[-1])
vertical_lines3 = vertical_lines3[-1:]
#print("horizontal : ", horizontal_lines3[0])
#print("vertical : ", vertical_lines3[0])

cv2.line(corner3, (horizontal_lines3[0][0], horizontal_lines3[0][1]), (horizontal_lines3[0][2], horizontal_lines3[0][3]), color=(0,255,0), thickness=1)
cv2.line(corner3, (vertical_lines3[0][0], vertical_lines3[0][1]), (vertical_lines3[0][2], vertical_lines3[0][3]), color=(0,255,0), thickness=1)

cv2.line(otsu3, (horizontal_lines3[0][0], horizontal_lines3[0][1]), (horizontal_lines3[0][2], horizontal_lines3[0][3]), (255,0,0), 1)
cv2.line(otsu3, (vertical_lines3[0][0], vertical_lines3[0][1]), (vertical_lines3[0][2], vertical_lines3[0][3]), (255,0,0), 1)

lines = cv2.HoughLines(otsu4, rho=1, theta=np.pi/180, threshold=10) 

horizontal_lines4 = []
vertical_lines4 = []
for i in range(lines.shape[0]):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 5000 * (-b)) - 2
    y1 = int(y0 + 5000 * a) - 2
    x2 = int(x0 - 5000 * (-b)) - 2
    y2 = int(y0 - 5000 * a) - 2
    #print(theta)
    if theta <= 0.1:
        vertical_lines4.append([x1, y1, x2, y2, rho])
    if theta >= 1.5 and theta <= 1.72:
        horizontal_lines4.append([x1, y1, x2, y2, rho])
        
horizontal_lines4.sort(key=lambda x: x[-1])
horizontal_lines4 = horizontal_lines4[-1:]
vertical_lines4.sort(key=lambda x: x[-1])
vertical_lines4 = vertical_lines4[-1:]
#print("horizontal : ", horizontal_lines4[0])
#print("vertical : ", vertical_lines4[0])

cv2.line(corner4, (horizontal_lines4[0][0], horizontal_lines4[0][1]), (horizontal_lines4[0][2], horizontal_lines4[0][3]), color=(0,255,0), thickness=1)
cv2.line(corner4, (vertical_lines4[0][0], vertical_lines4[0][1]), (vertical_lines4[0][2], vertical_lines4[0][3]), color=(0,255,0), thickness=1)

cv2.line(otsu4, (horizontal_lines4[0][0], horizontal_lines4[0][1]), (horizontal_lines4[0][2], horizontal_lines4[0][3]), (255,0,0), 1)
cv2.line(otsu4, (vertical_lines4[0][0], vertical_lines4[0][1]), (vertical_lines4[0][2], vertical_lines4[0][3]), (255,0,0), 1)
   

cv2.imshow("corner1", corner1)
cv2.imshow("corner2", corner2)
cv2.imshow("corner3", corner3)
cv2.imshow("corner4", corner4)
cv2.imshow("otsu corner1: " , otsu1)
cv2.imshow("otsu corner2: " , otsu2)
cv2.imshow("otsu corner3: " , otsu3)
cv2.imshow("otsu corner4: " , otsu4)

'''
cv2.imshow("corner4", corner4)

cv2.imshow("otsu corner1: " , otsu1)
cv2.imshow("otsu corner2: " , otsu2)
cv2.imshow("otsu corner3: " , otsu3)
cv2.imshow("otsu corner4: " , otsu4)
cv2.imshow("pixels1", otsu1)
cv2.imshow("Corners", stack)
cv2.imshow("Otsu Corners", stackOtsu)
'''

cv2.waitKey(0)
