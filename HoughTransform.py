import cv2
import numpy as np

img = cv2.imread('baseball_card7.png', 0)
black_img = np.zeros(img.shape)
edges = cv2.Canny(img, 100, 150)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=60)
N = lines.shape[0]

horizontal_lines = []
vertical_lines = []
for i in range(N):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    if y1 == y2 or abs(y1 - y2) <= 20:
        horizontal_lines.append(lines[i])
    if x1 == x2 or abs(x1 - x2) <= 20:
        vertical_lines.append(lines[i])
    # cv2.line(black_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

vertical_lines.sort(key=lambda x: x[0][0])
vertical_lines = vertical_lines[:2] + vertical_lines[-2:]

length_bottom = vertical_lines[1][0][0] - vertical_lines[0][0][0]
length_top = vertical_lines[3][0][0] - vertical_lines[2][0][0]
print(length_bottom, length_top)
print("Bottom to Top Ratio ", length_bottom / length_top)

horizontal_lines.sort(key=lambda x: x[0][0])
horizontal_lines = horizontal_lines[:2] + horizontal_lines[-2:]

length_left = horizontal_lines[1][0][0] - horizontal_lines[0][0][0]
length_right = horizontal_lines[3][0][0] - horizontal_lines[2][0][0]
print(length_right, length_left)
print("Left to Right Ratio ", length_left / length_right)

borders = horizontal_lines + vertical_lines
for line in borders:
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(black_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

img = img.astype(np.float32) / 255
stack = np.hstack((img, black_img))
cv2.imshow('Hough Lines', stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
