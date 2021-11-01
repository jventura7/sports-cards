import cv2
import numpy as np

img = cv2.imread('baseball_card.jpg', 0)
img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
black_img = np.zeros(img.shape)
edges = cv2.Canny(img, 100, 150)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
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

vertical_lines.sort(key=lambda x: x[0][0])
vertical_lines = vertical_lines[:2] + vertical_lines[-2:]

horizontal_lines.sort(key=lambda x: x[0][0])
horizontal_lines = horizontal_lines[:2] + horizontal_lines[-2:]

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

cv2.imwrite('houghlines.jpg', black_img)
cv2.imshow('Canny', edges)
cv2.imshow('OG', img)
cv2.imshow('Hough Lines', black_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
