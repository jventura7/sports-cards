import cv2
import numpy as np

img = cv2.imread('baseball_card.jpg', 0)
black_img = np.zeros(img.shape)
edges = cv2.Canny(img, 550, 600)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=10, maxLineGap=300)
N = lines.shape[0]
N = 8
for i in range(N):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]
    cv2.line(black_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imwrite('houghlines.jpg', black_img)
cv2.imshow('Canny', edges)
cv2.imshow('OG', img)
cv2.imshow('Hough Lines', black_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)