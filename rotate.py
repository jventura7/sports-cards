import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel
from matplotlib import cm

'''
img = cv2.imread('baseball_card', 0)
threshold = threshold_otsu(img)
binary = image < threshold
'''


img = cv2.imread('baseball_card_2.png', 0)
plt.imshow(img, cmap = plt.cm.gray)
plt.axis('off')
plt.title('Original Image')
plt.show()
edges = cv2.Canny(img, 100, 120)


#edges = sobel(binary)
h, theta, d = hough_line(edges)
accum, angles, dists = hough_line_peaks(h, theta, d)
angle = np.rad2deg(mode(angles)[0][0])
if (angle < 0):
    angle = angle + 90
else:
    angle = angle - 90
fixed = rotate(img, angle)
plt.imshow(fixed, cmap = plt.cm.gray)
plt.axis('off')
plt.title('Rotated Image')
plt.savefig(r'C:\Users\rajan\OneDrive - Virginia Tech\Fourth Year\Senior Design\sports-cards\fixed_image.png')
plt.show()