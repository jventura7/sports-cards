import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm
import matplotlib.pyplot as plt

# Read image and perform Canny Edge Detection
img = cv2.imread('baseball_card.jpg', 0)
black_img = np.zeros(img.shape)
edges = cv2.Canny(img, 100, 120)

# Obtain Hough Lines
h, theta, d = hough_line(edges)
fig, ax = plt.subplots()
ax.imshow(img, cmap=cm.gray)
row1, col1 = img.shape
origin = np.array((0, img.shape[1]))

horizontal_lines = []
vertical_lines = []

for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.5*h.max())):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    print(y0, y1)
    if (y0 == y1 or abs(y0 - y1) <= 20) and y0 <= 600 and y1 <= 600:
        horizontal_lines.append([y0, y1])
    if abs(y0) >= 10000 and abs(y1) >= 10000:
        vertical_lines.append([y0, y1])
    # ax.plot(origin, (y0, y1), '-r')
# ax.axis((0, col1, row1, 0))
# ax.set_title('Detected lines')
# ax.set_axis_off()

# Sort Hough Lines to get card borders
for i in range(len(vertical_lines)):
    for j in range(len(vertical_lines[i])):
        vertical_lines[i] = [abs(vertical_lines[i][0]), vertical_lines[i][1]]
        break
vertical_lines.sort(key=lambda x: x[0])
vertical_lines = vertical_lines[:2] + vertical_lines[-2:]
print(vertical_lines)
length_left = vertical_lines[1][0] - vertical_lines[0][0]
length_right = vertical_lines[3][0] - vertical_lines[2][0]
total_left_right = length_left + length_right
print("Length of margins (pixels): Left: {left}  Right: {right}".format(left=int(length_left), right=int(length_right)))
print("Left to Right Ratio: {left_ratio}:{right_ratio}".format(left_ratio=int(round(length_left / total_left_right * 100)),
                                                               right_ratio=int(round(length_right / total_left_right * 100))))

horizontal_lines.sort(key=lambda x: x[0])
horizontal_lines = horizontal_lines[:2] + horizontal_lines[-2:]

length_bottom = horizontal_lines[1][0] - horizontal_lines[0][0]
length_top = horizontal_lines[3][0] - horizontal_lines[2][0]
total_bottom_top = length_bottom + length_top
print("Length of margins (pixels): Bottom: {bottom}  Top: {top}".format(bottom=int(length_bottom), top=int(length_top)))
print("Bottom to Top Ratio: {bottom_ratio}:{top_ratio}".format(bottom_ratio=int(round(length_bottom / total_bottom_top * 100)),
                                                               top_ratio=int(round(length_top / total_bottom_top * 100))))
# Plot card borders
borders = horizontal_lines + vertical_lines
for border in borders:
    ax.plot(origin, (border[0], border[1]), '-r')
ax.axis((0, col1, row1, 0))
ax.set_title('Detected lines')
ax.set_axis_off()
fig.savefig('houghlines.png')

img = img.astype(np.float32) / 255
stack = np.hstack((img, black_img))
cv2.imshow('Hough Lines', stack)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
