import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2


def colorFit(value, palette):
    return palette[np.argmin(np.abs(palette - value))]


def neighbour(img, N):
    if (len(img.shape) != 3):
        L1 = img.shape[0]
        L2 = img.shape[1]
        K1 = np.round(np.linspace(
            0, L1-1, np.round(N*L1).astype(int))).astype(int)
        K2 = np.round(np.linspace(
            0, L2-1, np.round(N*L2).astype(int))).astype(int)
        lenK1 = len(K1)
        lenK2 = len(K2)
        img2 = np.zeros((lenK1, lenK2))
        for i in range(lenK1):
            for j in range(lenK2):
                img2[i, j] = img[K1[i], K2[j]]
        return img2
    else:
        L1 = img.shape[0]
        L2 = img.shape[1]
        K1 = np.round(np.linspace(
            0, L1-1, np.round(N*L1).astype(int))).astype(int)
        K2 = np.round(np.linspace(
            0, L2-1, np.round(N*L2).astype(int))).astype(int)
        lenK1 = len(K1)
        lenK2 = len(K2)
        img2 = np.zeros((lenK1, lenK2, 3))
        for i in range(lenK1):
            for j in range(lenK2):
                for k in range(3):
                    img2[i, j, k] = img[K1[i], K2[j], k]
        return img2


def interpolation(img, N):
    if (len(img.shape) != 3):
        L1 = img.shape[0]
        L2 = img.shape[1]
        K1 = np.round(np.linspace(
            0, L1-1, np.round(N*L1).astype(int))).astype(int)
        K2 = np.round(np.linspace(
            0, L2-1, np.round(N*L2).astype(int))).astype(int)
        lenK1 = len(K1)
        lenK2 = len(K2)
        img2 = np.zeros((lenK1, lenK2))
        for i in range(lenK1):
            for j in range(lenK2):
                img2[i, j] = img[K1[i], K2[j]]
        return img2
    else:
        L1 = img.shape[0]
        L2 = img.shape[1]
        N1 = np.linspace(0, L1-1, L1)
        N2 = np.linspace(0, L2-1, L2)
        K1 = np.round(np.linspace(
            0, L1-1, np.round(N*L1).astype(int))).astype(int)
        K2 = np.round(np.linspace(
            0, L2-1, np.round(N*L2).astype(int))).astype(int)
        f1 = interpolate.RectBivariateSpline(N1, N2, img[:, :, 0])
        f2 = interpolate.RectBivariateSpline(N1, N2, img[:, :, 1])
        f3 = interpolate.RectBivariateSpline(N1, N2, img[:, :, 2])
        lenK1 = len(K1)
        lenK2 = len(K2)
        img2 = np.zeros((lenK1, lenK2, 3))
        for i in range(lenK1):
            for j in range(lenK2):
                img2[i, j, 0] = f1(K1[i], K2[j])
                img2[i, j, 1] = f2(K1[i], K2[j])
                img2[i, j, 2] = f3(K1[i], K2[j])
        return img2


def calcMedian(img, N):
    L1 = img.shape[0]
    L2 = img.shape[1]
    K1 = np.round(np.linspace(0, L1-1, np.round(N*L1).astype(int))).astype(int)
    K2 = np.round(np.linspace(0, L2-1, np.round(N*L2).astype(int))).astype(int)
    if (len(img.shape) != 3):
        img2 = np.zeros((len(K1), len(K2)))
        for i in range(1, len(K1)-1):
            for j in range(1, len(K2)-1):
                tab1 = []
                tab1.append(img[K1[i-1], K2[j-1], 0])
                tab1.append(img[K1[i-1], K2[j], 0])
                tab1.append(img[K1[i-1], K2[j+1], 0])
                tab1.append(img[K1[i], K2[j-1], 0])
                tab1.append(img[K1[i], K2[j+1], 0])
                tab1.append(img[K1[i+1], K2[j-1], 0])
                tab1.append(img[K1[i+1], K2[j], 0])
                tab1.append(img[K1[i+1], K2[j+1], 0])

                img2[i, j] = np.median(tab1)
        for j in range(0, len(K2)):
            img2[0][j] = img2[1][j]
            img2[len(K1)-1][j] = img2[len(K1)-2][j]
        for i in range(0, len(K1)):
            img2[i][0] = img2[i][1]
            img2[i][len(K2)-1] = img2[i][len(K2)-2]
    else:
        img2 = np.zeros((len(K1), len(K2), 3))
        for i in range(1, len(K1)-1):
            for j in range(1, len(K2)-1):
                tab1 = []
                tab1.append(img[K1[i-1], K2[j-1], 0])
                tab1.append(img[K1[i-1], K2[j], 0])
                tab1.append(img[K1[i-1], K2[j+1], 0])
                tab1.append(img[K1[i], K2[j-1], 0])
                tab1.append(img[K1[i], K2[j+1], 0])
                tab1.append(img[K1[i+1], K2[j-1], 0])
                tab1.append(img[K1[i+1], K2[j], 0])
                tab1.append(img[K1[i+1], K2[j+1], 0])
                tab2 = []
                tab2.append(img[K1[i-1], K2[j-1], 1])
                tab2.append(img[K1[i-1], K2[j], 1])
                tab2.append(img[K1[i-1], K2[j+1], 1])
                tab2.append(img[K1[i], K2[j-1], 1])
                tab2.append(img[K1[i], K2[j+1], 1])
                tab2.append(img[K1[i+1], K2[j-1], 1])
                tab2.append(img[K1[i+1], K2[j], 1])
                tab2.append(img[K1[i+1], K2[j+1], 1])
                tab3 = []
                tab3.append(img[K1[i-1], K2[j-1], 2])
                tab3.append(img[K1[i-1], K2[j], 2])
                tab3.append(img[K1[i-1], K2[j+1], 2])
                tab3.append(img[K1[i], K2[j-1], 2])
                tab3.append(img[K1[i], K2[j+1], 2])
                tab3.append(img[K1[i+1], K2[j-1], 2])
                tab3.append(img[K1[i+1], K2[j], 2])
                tab3.append(img[K1[i+1], K2[j+1], 2])
                img2[i, j, 0] = np.median(tab1)
                img2[i, j, 1] = np.median(tab2)
                img2[i, j, 2] = np.median(tab3)
        for j in range(0, len(K2)):
            img2[0][j][0] = img2[1][j][0]
            img2[0][j][1] = img2[1][j][1]
            img2[0][j][2] = img2[1][j][2]
            img2[len(K1)-1][j][0] = img2[len(K1)-2][j][0]
            img2[len(K1)-1][j][1] = img2[len(K1)-2][j][1]
            img2[len(K1)-1][j][2] = img2[len(K1)-2][j][2]
        for i in range(0, len(K1)):
            img2[i][0][0] = img2[i][1][0]
            img2[i][0][1] = img2[i][1][1]
            img2[i][0][2] = img2[i][1][2]
            img2[i][len(K2)-1][0] = img2[i][len(K2)-2][0]
            img2[i][len(K2)-1][1] = img2[i][len(K2)-2][1]
            img2[i][len(K2)-1][2] = img2[i][len(K2)-2][2]

    return img2


def processSmallPhotos():
    imagesList = ['0014.jpg', '0013.jpg', '0008.png']
    xList = [(25, 40), (20, 50), (50, 100)]
    yList = [(30, 55), (20, 50), (100, 150)]
    for i in range(len(imagesList)):
        plt.figure()
        img = plt.imread(imagesList[i])
        if img.dtype == "uint8":
            img = img.astype(float)/255
        plt.subplot(251), plt.imshow(img)
        plt.title("Obraz oryginalny")
        img2 = img[xList[i][0]:xList[i][1], yList[i][0]:yList[i][1], :]
        plt.subplot(252), plt.imshow(img2)
        plt.title("Wycinek obrazu")
        img3 = neighbour(img2, 2)
        plt.subplot(253), plt.imshow(img3)
        plt.title("Metoda najblizszego sasiada: N=2")
        img4 = interpolation(img2, 2)
        plt.subplot(254), plt.imshow(img4)
        plt.title("Metoda interpolacji: N=2")

        img42 = calcMedian(img2, 0.6)
        plt.subplot(255), plt.imshow(img42)
        plt.title("Metoda mediany: N=0.6")

        # Wykrywanie krawedzi

        imgUINT = np.uint8(img2*255)
        img6 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(256), plt.imshow(img6)
        plt.title("Krawędzie - wycinek obrazu")

        imgUINT = np.uint8(img3*255)
        img7 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(257), plt.imshow(img7)
        plt.title("Krawędzie - metoda najbliższych sąsiadów")

        imgUINT = np.uint8(img4*255)
        img8 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(258), plt.imshow(img8)
        plt.title("Krawędzie - metoda interpolacji")

        imgUINT = np.uint8(img42*255)
        img9 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(259), plt.imshow(img9)
        plt.title("Krawędzie - metoda mediany")
    plt.show()


def processBigPhotos():
    imagesList = ['0001.jpg', '0003.jpg']
    xList = [(800, 1100), (2000, 2300)]
    yList = [(800, 1100), (1500, 1800)]
    for i in range(len(imagesList)):
        plt.figure()
        img = plt.imread(imagesList[i])
        if img.dtype == "uint8":
            img = img.astype(float)/255
        plt.subplot(251), plt.imshow(img)
        plt.title("Obraz oryginalny")
        img2 = img[xList[i][0]:xList[i][1], yList[i][0]:yList[i][1], :]
        plt.subplot(252), plt.imshow(img2)
        plt.title("Wycinek obrazu")
        img3 = neighbour(img2, 0.05)
        plt.subplot(253), plt.imshow(img3)
        plt.title("Metoda najblizszego sasiada: N=0.05")
        img4 = interpolation(img2, 0.05)
        plt.subplot(254), plt.imshow(img4)
        plt.title("Metoda interpolacji: N=0.05")

        img42 = calcMedian(img2, 0.05)
        plt.subplot(255), plt.imshow(img42)
        plt.title("Metoda mediany: N=0.05")

        # Wykrywanie krawedzi

        imgUINT = np.uint8(img2*255)
        img6 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(256), plt.imshow(img6)
        plt.title("Krawędzie - wycinek obrazu")

        imgUINT = np.uint8(img3*255)
        img7 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(257), plt.imshow(img7)
        plt.title("Krawędzie - metoda najbliższych sąsiadów")

        imgUINT = np.uint8(img4*255)
        img8 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(258), plt.imshow(img8)
        plt.title("Krawędzie - metoda interpolacji")

        imgUINT = np.uint8(img42*255)
        img9 = cv2.Canny(imgUINT, 100, 200)
        plt.subplot(259), plt.imshow(img9)
        plt.title("Krawędzie - metoda mediany")
    plt.show()


def dithFloydSteinberg():
    imagesList = ['0009.png', '0008.png', '0013.jpg']
    numberOfBits = [1, 2, 4]
    for i in range(len(imagesList)):
        plt.figure()
        img = plt.imread(imagesList[i])
        if img.dtype == "uint8":
            img = img.astype(float)/255
        img = 0.2126 * img[:, :, 0] + 0.7152 * \
            img[:, :, 1] + 0.0722 * img[:, :, 2]
        plt.subplot(221), plt.imshow(img, cmap=plt.cm.gray)
        plt.title("Obraz oryginalny")
        for j in range(len(numberOfBits)):
            palette = np.linspace(0, 1, 2**numberOfBits[j])
            a = img.shape[0]
            b = img.shape[1]
            img2 = img.copy()
            for y in range(0, b):
                for x in range(0, a):
                    oldpixel = np.copy(img2[x][y])
                    newpixel = colorFit(oldpixel, palette)
                    img2[x][y] = newpixel
                    quant_error = oldpixel - newpixel
                    if x+1 < a:
                        img2[x + 1, y] = img2[x + 1, y] + quant_error * 7 / 16
                    if y+1 < b and x-1 > 0:
                        img2[x - 1, y + 1] = img2[x - 1, y + 1] + \
                            quant_error * 3 / 16
                    if y+1 < b:
                        img2[x, y + 1] = img2[x, y + 1] + quant_error * 5 / 16
                    if x+1 < a and y+1 < b:
                        img2[x + 1, y + 1] = img2[x + 1, y + 1] + \
                            quant_error * 1 / 16
                    numb = 222+j
            plt.subplot(numb), plt.imshow(img2, cmap=plt.cm.gray)
            plt.title("Dithering Floyd - Steingerga dla " +
                      str(numberOfBits[j])+" bitów")
    plt.show()


processSmallPhotos()
processBigPhotos()
dithFloydSteinberg()
