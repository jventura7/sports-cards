import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


pull_img = "image5.png"
new_img = "resized_image5.png"

image = Image.open(pull_img)
resized_img = image.resize((2000, 1000))
resized_img.save(new_img)

# Read image and perform Canny Edge Detection
img = cv2.imread(new_img, 0)
'''
black_img = np.zeros(img.shape)
edges = cv2.Canny(img, 100, 120)
img = cv2.imread('image1.png', 0)
'''
black_img = np.zeros(img.shape)
edges = cv2.Canny(img, 100, 150)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
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
'''
print("Top left corner: (" + str(vertical_lines[0][0][0]) + ", " + str(horizontal_lines[0][0][0]) + ")")
print("Bottom left corner: (" + str(vertical_lines[0][0][0]) + ", " + str(horizontal_lines[3][0][0]) + ")")
print("Top right corner: (" + str(vertical_lines[3][0][0]) + ", " + str(horizontal_lines[0][0][0]) + ")")
print("Bottom right corner: (" + str(vertical_lines[3][0][0]) + ", " + str(horizontal_lines[3][0][0]) + ")")

print()

left_x = (vertical_lines[0][0][0] + vertical_lines[1][0][0]) / 2
right_x = (vertical_lines[2][0][0] + vertical_lines[3][0][0]) / 2

top_y = (horizontal_lines[0][0][0] + horizontal_lines[1][0][0]) / 2
bottom_y = (horizontal_lines[2][0][0] + horizontal_lines[3][0][0]) / 2

print("Top left center point: (" + str(left_x) + ", " + str(top_y) + ")")
print("Bottom left center point: (" + str(left_x) + ", " + str(bottom_y) + ")")
print("Top right center point: (" + str(right_x) + ", " + str(top_y) + ")")
print("Bottom right center point: (" + str(right_x) + ", " + str(bottom_y) + ")")


ax2.add_patch(patches.Rectangle((left_x, top_y), vertical_lines[0][0][0] - left_x, horizontal_lines[0][0][0] - top_y, linewidth=1, edgecolor='r', facecolor='none'))
ax2.add_patch(patches.Rectangle((left_x, bottom_y), vertical_lines[0][0][0] - left_x, horizontal_lines[3][0][0] - bottom_y, linewidth=1, edgecolor='r', facecolor='none'))
ax2.add_patch(patches.Rectangle((right_x, top_y), vertical_lines[3][0][0] - right_x, horizontal_lines[0][0][0] - top_y, linewidth=1, edgecolor='r', facecolor='none'))
ax2.add_patch(patches.Rectangle((right_x, bottom_y), vertical_lines[3][0][0] - right_x, horizontal_lines[3][0][0] - bottom_y, linewidth=1, edgecolor='r', facecolor='none'))

plt.show()

'''import cv2
import numpy as np
import matplotlib.pyplot as plt

def shi_tomashi(image):
    """
    Use Shi-Tomashi algorithm to detect corners
    Args:
        image: np.array
    Returns:
        corners: list
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    print('\nThe corner points are...\n')

    im = image.copy()
    for index, c in enumerate(corners):
        x, y = c
        cv2.circle(im, (x, y), 3, 255, -1)
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(im, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    plt.figure(1)
    plt.imshow(im)
    plt.title('Corner Detection: Shi-Tomashi')
    plt.show()
    return corners

img_name = "image4.png"

# Load image, convert to grayscale, and find edges
image = cv2.imread(img_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

# Find contour and sort by contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
ROI = image
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = image
    ROI[y:y+h, x:x+w] = 0
    #ROI = image[y-50:y+h+50, x-50:x+w+50]
    break
ROI[ROI != 0] = 255

output = shi_tomashi(ROI)

corners = cv2.imread(img_name)#[y-25:y+h+25, x-25:x+w+25]
cv2.rectangle(corners, (output[0][0], output[0][1]), (output[0][0] + 50, output[0][1] + 50), (255,0,0), 5)
cv2.rectangle(corners, (output[1][0], output[1][1]), (output[1][0] + 50, output[1][1] - 50), (255,0,0), 5)
cv2.rectangle(corners, (output[2][0], output[2][1]), (output[2][0] - 50, output[2][1] + 50), (255,0,0), 5)
cv2.rectangle(corners, (output[3][0], output[3][1]), (output[3][0] - 50, output[3][1] - 50), (255,0,0), 5)

plt.figure(1)
plt.imshow(corners)
plt.title('Corner Detection: Shi-Tomashi')
plt.show()
'''