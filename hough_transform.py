import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def detectVerticalLines(img, test_threshold):
    median = np.median(img)
    edges = cv2.Canny(img, 0.65 * median, 0.8 * median)
    cv2.imshow('canny', edges)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=test_threshold)
    N = lines.shape[0]

    # marks x positions of vertical lines
    seen = set()
    rightTwo = set()
    leftTwo = set()
    strongest = 0
    dontInclude = False
    vertical_lines = []
    row1, col1 = img.shape
    for i in range(N):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        x1 = int(x0 + 5000 * (-b))
        x2 = int(x0 - 5000 * (-b))
        if x1 == x2 or abs(x1 - x2) <= 160:
            for x in seen:
                if abs(x - x1) < 40:
                    dontInclude = True
            if (len(leftTwo) == 2 and x1 < 400) or (len(rightTwo) == 2 and x1 > 1600):
                dontInclude = True
            if 1600 > x1 > 400:
                dontInclude = True

            if not dontInclude:
                strongest += 1
                vertical_lines.append(lines[i])
                seen.add(x1)
                if x1 < 400:
                    leftTwo.add(x1)
                if x1 > 1600:
                    rightTwo.add(x1)

            if strongest == 4:
                break
        dontInclude = False

    return vertical_lines, img, row1, col1

def detectHorizontalLines(img, test_threshold):
    median = np.median(img)
    edges = cv2.Canny(img, 0.65 * median, 0.8 * median)
    cv2.imshow('canny hor', edges)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=test_threshold)
    N = lines.shape[0]

    # mark y positions of horizontal lines
    seen = set()
    bottomTwo = set()
    topTwo = set()
    strongest = 0
    horizontal_lines = []
    row1, col1 = img.shape
    dontInclude = False
    for i in range(N):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        y0 = b * rho
        y1 = int(y0 + 5000 * a)
        y2 = int(y0 - 5000 * a)
        if y1 == y2 or abs(y1 - y2) <= 160:
            for y in seen:
                if abs(y - y1) < 40:
                    dontInclude = True
            if (len(bottomTwo) == 2 and y1 > 2400) or (len(topTwo) == 2 and y1 < 800):
                dontInclude = True
            if 2400 > y1 > 800:
                dontInclude = True

            if not dontInclude:
                strongest += 1
                horizontal_lines.append(lines[i])
                seen.add(y1)
                if y1 > 2400:
                    bottomTwo.add(y1)
                if y1 < 800:
                    topTwo.add(y1)
            if strongest == 4:
                break
        dontInclude = False

    return horizontal_lines, img, row1, col1

def displayHorizontalLines(img, horizontal):
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
    ax2.set_title('Detected Horizontal Lines')

def displayVerticalLines(img, vertical):
    i, outer, inner, rotation = 0, 0, 0, 0
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
        if i == 0:
            outer = math.atan2(abs(x2 - x1), abs(y2 - y1))
        elif i == 1:
            inner = math.atan2(abs(x2 - x1), abs(y2 - y1))
            rotation = math.degrees(abs(inner - outer))
        ax3.plot((x1, x2), (y1, y2), 'red')
        i += 1
    print("The rotation within the card is " + str(rotation) + "degrees.")
    rows, cols = img.shape
    ax3.axis((0, cols, rows, 0))
    ax3.set_title('Detected Vertical Lines')
    return rotation

def detectFinalMargins(img, vertical_lines, horizontal_lines):
    vertical_lines = vertical_lines[:2] + vertical_lines[-2:]

    length_bottom = vertical_lines[1][0][0] - vertical_lines[0][0][0]
    length_top = vertical_lines[3][0][0] - vertical_lines[2][0][0]
    print("Left:", length_bottom, "Right:", length_top)
    print("Left to Right Ratio ", int(round((length_bottom / (length_bottom + length_top)) * 100)), ":",
          int(round((length_top / (length_top + length_bottom)) * 100)))

    horizontalRatio = int(round((length_bottom / (length_bottom + length_top)) * 100))

    horizontal_lines = horizontal_lines[:2] + horizontal_lines[-2:]
    length_left = horizontal_lines[1][0][0] - horizontal_lines[0][0][0]
    length_right = horizontal_lines[3][0][0] - horizontal_lines[2][0][0]
    print("Bottom:", length_right, "Top: ", length_left)
    print("Bottom to Top Ratio ", int(round((length_left / (length_left + length_right)) * 100)), ":",
          int(round((length_right / (length_left + length_right)) * 100)))

    verticalRatio = int(round((length_left / (length_left + length_right)) * 100))

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
    row1, col1 = img.shape
    ax2.axis((0, col1, row1, 0))
    ax2.set_title('Corner Detection')
    ax2.set_axis_off()
    return vertical_lines, horizontal_lines, ax2, horizontalRatio, verticalRatio

def detectCorners(img, vertical_lines, horizontal_lines, ax2):
    left_x = (vertical_lines[0][0][0] + vertical_lines[1][0][0]) // 2
    right_x = (vertical_lines[2][0][0] + vertical_lines[3][0][0]) // 2

    top_y = (horizontal_lines[0][0][0] + horizontal_lines[1][0][0]) // 2
    bottom_y = (horizontal_lines[2][0][0] + horizontal_lines[3][0][0]) // 2

    ax2.add_patch(
        patches.Rectangle((left_x, top_y), vertical_lines[0][0][0] - left_x, horizontal_lines[0][0][0] - top_y,
                          linewidth=1,
                          edgecolor='r', facecolor='none'))
    ax2.add_patch(
        patches.Rectangle((left_x, bottom_y), vertical_lines[0][0][0] - left_x, horizontal_lines[3][0][0] - bottom_y,
                          linewidth=1, edgecolor='r', facecolor='none'))
    ax2.add_patch(
        patches.Rectangle((right_x, top_y), vertical_lines[3][0][0] - right_x, horizontal_lines[0][0][0] - top_y,
                          linewidth=1, edgecolor='r', facecolor='none'))
    ax2.add_patch(
        patches.Rectangle((right_x, bottom_y), vertical_lines[3][0][0] - right_x, horizontal_lines[3][0][0] - bottom_y,
                          linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()


    corner1 = img[int(horizontal_lines[0][0][0]):int(top_y), int(vertical_lines[0][0][0]):int(left_x)]
    corner2 = img[int(bottom_y):int(horizontal_lines[3][0][0]), int(vertical_lines[0][0][0]):int(left_x)]
    corner3 = img[int(horizontal_lines[0][0][0]):int(top_y), int(right_x):int(vertical_lines[3][0][0])]
    corner4 = img[int(bottom_y):int(horizontal_lines[3][0][0]), int(right_x):int(vertical_lines[3][0][0])]

    ret1, otsu1 = cv2.threshold(corner1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, otsu2 = cv2.threshold(corner2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, otsu3 = cv2.threshold(corner3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret4, otsu4 = cv2.threshold(corner4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

    corners = [corner1, corner2, corner3, corner4]
    cornerPixels = [cornerPixels1, cornerPixels2, cornerPixels3, cornerPixels4]

    printCorners(corners, cornerPixels)
    return "{:.2f}".format(np.std(cornerPixels1)), "{:.2f}".format(np.std(cornerPixels2)), "{:.2f}".format(np.std(cornerPixels3)), "{:.2f}".format(np.std(cornerPixels4))

def printCorners(corners, pixels):
    corner1, corner2, corner3, corner4 = corners
    cornerPixels1, cornerPixels2, cornerPixels3, cornerPixels4 = pixels
    print("CORNERS DETECTED USING HOUGH TRANSFORM TECHNIQUE")
    print("Top Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(corner1))), "\t", "Standard Deviation: ",
          str("{:.2f}".format(np.std(corner1))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner1))))
    print("Bottom Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(corner2))), "\t", "Standard Deviation: ",
          str("{:.2f}".format(np.std(corner2))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner2))))
    print("Top Right Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(corner3))), "\t", "Standard Deviation: ",
          str("{:.2f}".format(np.std(corner3))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner3))))
    print("Bottom Right Corner:\t", "Average: ", str("{:.2f}".format(np.mean(corner4))), "\t", "Standard Deviation: ",
          str("{:.2f}".format(np.std(corner4))), "\t", "Variance: ", str("{:.2f}".format(np.var(corner4))))

    print("\nBELOW ARE CORNERS USING OTSU METHOD")
    print("Top Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels1))), "\t",
          "Standard Deviation: ",
          str("{:.2f}".format(np.std(cornerPixels1))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels1))))
    print("Bottom Left Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels2))), "\t",
          "Standard Deviation: ", str("{:.2f}".format(np.std(cornerPixels2))), "\t", "Variance: ",
          str("{:.2f}".format(np.var(cornerPixels2))))
    print("Top Right Corner:\t\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels3))), "\t",
          "Standard Deviation: ",
          str("{:.2f}".format(np.std(cornerPixels3))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels3))))
    print("Bottom Right Corner:\t", "Average: ", str("{:.2f}".format(np.mean(cornerPixels4))), "\t",
          "Standard Deviation: ",
          str("{:.2f}".format(np.std(cornerPixels4))), "\t", "Variance: ", str("{:.2f}".format(np.var(cornerPixels4))))


# Import image
def test(img):
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (2240, 3120))

    horizontalThreshold = 40
    verticalThreshold = 40
    horizontal_lines, _, row1, col1 = detectHorizontalLines(img, horizontalThreshold)
    vertical_lines, _, row1, col1 = detectVerticalLines(img, verticalThreshold)
    vertical, horizontal, ax2, verticalRatio, horizontalRatio = detectFinalMargins(img, vertical_lines, horizontal_lines)

    rotate = displayVerticalLines(img, vertical_lines)
    displayHorizontalLines(img, horizontal_lines)
    c1, c2, c3, c4 = detectCorners(img, vertical, horizontal, ax2)
    cv2.waitKey(0)
    return verticalRatio, horizontalRatio, c1, c2, c3, c4, rotate

# works well
# test("psa10_152.jpg")
test("psa5_126.png")
# test("psa4_98.png")
# does not
# test("no_slab2.jpg")
