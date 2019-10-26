import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read image, make gray scale, threshold to binary
img = cv.imread('mouse_brain-one_FOV.tif')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

plt.figure("Watershed Segmentation")
# plt.subplot(1, 6, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.subplot(1, 6, 2)
# plt.imshow(thresh)
# plt.title('Binarized/threshold')

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
# plt.subplot(1, 4, 1)
# plt.imshow(opening)
# plt.title('Morphological Opening')

# finding sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
plt.subplot(2, 3, 1)
plt.imshow(sure_bg)
plt.title('Dilation:sure_bg')

# distance transform to find sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
print("dist_transform is: ", dist_transform.max())
ret, sure_fg = cv.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
plt.subplot(2, 3, 2)
plt.imshow(dist_transform)
plt.title('dist transform')
plt.subplot(2, 3, 3)
plt.imshow(sure_fg)
plt.title('thresholds: sure_fg')

# Finding unknown region
sure_fg = np.uint8(sure_fg)
print(sure_fg.dtype)
print(sure_bg.dtype)
unknown = cv.subtract(sure_bg, sure_fg)
plt.subplot(2, 3, 4)
plt.imshow(unknown)
plt.title('unknown')

# mark Labeling
ret, markers = cv.connectedComponents(sure_fg)
# Adding one to sure background, so it is 1 instead of 0
markers = markers + 1
# marking the unknown region as zero
markers[unknown == 255] = 0
markers = cv.watershed(img, markers)
plt.subplot(2, 3, 5)
plt.imshow(markers)
plt.title('markers')

img[markers == -1] = [255, 0, 0]
plt.subplot(2, 3, 6)
plt.imshow(img)
plt.title('Marked Image post segmentation')