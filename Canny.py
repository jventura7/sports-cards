import cv2

img = cv2.imread('baseball_card.jpg', 0)
edges = cv2.Canny(img, 650, 630)
cv2.imshow('Original', img)
cv2.imshow('Canny', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)