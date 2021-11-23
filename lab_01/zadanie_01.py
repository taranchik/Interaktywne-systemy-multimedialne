import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = plt.imread('./src/pic1.png')

# print(img1.dtype)
# print(img1.shape)

img2 = cv2.imread('./src/pic2.jpg')

# print(img2.dtype)
# print(img2.shape)

# plt.imshow(img1)
# plt.show()

# COLOR_R = img1[:, :, 0]
# plt.imshow(R)
# plt.show()

# plt.imshow(R, cmap=plt.cm.gray)

img_GRAY = cv2.cvtColor(img2, cv2.cv2.COLOR_BGR2GRAY)
img_RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

GRAY_R = img_RGB[:, :, 0:1].copy()
GRAY_G = img_RGB[:, :, 1:2].copy()
GRAY_B = img_RGB[:, :, 2:].copy()

img_COLOR_R_ZERO = img_RGB.copy()
img_COLOR_R_ZERO[:, :, 1] = 0
img_COLOR_R_ZERO[:, :, 2] = 0

img_COLOR_G_ZERO = img_RGB.copy()
img_COLOR_G_ZERO[:, :, 0] = 0
img_COLOR_G_ZERO[:, :, 2] = 0

img_COLOR_B_ZERO = img_RGB.copy()
img_COLOR_B_ZERO[:, :, 0] = 0
img_COLOR_B_ZERO[:, :, 1] = 0

Y1 = 0.299 * GRAY_R + 0.587 * GRAY_G + 0.114 * GRAY_B
Y2 = 0.2126 * GRAY_R + 0.7152 * GRAY_G + 0.0722 * GRAY_B

plt.subplot(3, 3, 1)
plt.imshow(img_RGB)
plt.subplot(3, 3, 2)
plt.imshow(Y1, cmap=plt.cm.gray)
plt.subplot(3, 3, 3)
plt.imshow(Y2, cmap=plt.cm.gray)

plt.subplot(3, 3, 4)
plt.imshow(GRAY_R, cmap=plt.cm.gray)
plt.subplot(3, 3, 5)
plt.imshow(GRAY_G, cmap=plt.cm.gray)
plt.subplot(3, 3, 6)
plt.imshow(GRAY_B, cmap=plt.cm.gray)

plt.subplot(3, 3, 7)
plt.imshow(img_COLOR_R_ZERO)
plt.subplot(3, 3, 8)
plt.imshow(img_COLOR_G_ZERO)
plt.subplot(3, 3, 9)
plt.imshow(img_COLOR_B_ZERO)

plt.show()

fragment = img_GRAY[0:99, 0:99].copy()
plt.imshow(fragment)
plt.show()
