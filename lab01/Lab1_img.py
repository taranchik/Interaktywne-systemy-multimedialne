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

COLOR_R = img_RGB[:, :, 0:1].copy()
COLOR_G = img_RGB[:, :, 1:2].copy()
COLOR_B = img_RGB[:, :, 2:].copy()

GRAY_R = img_GRAY[:, :, 0:1].copy()
GRAY_G = img_GRAY[:, :, 1:2].copy()
GRAY_B = img_GRAY[:, :, 2:].copy()

print('img_RGB', img_RGB)
print('COLOR_R', COLOR_R)
print('COLOR_G', COLOR_G)
print('COLOR_B', COLOR_B)

Y1 = 0.299 * COLOR_R + 0.587 * COLOR_G + 0.114 * COLOR_B
Y2 = 0.2126 * COLOR_R + 0.7152 * COLOR_G + 0.0722 * COLOR_B

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
plt.imshow(COLOR_R)
plt.subplot(3, 3, 8)
plt.imshow(COLOR_G)
plt.subplot(3, 3, 9)
plt.imshow(COLOR_B)

plt.show()
