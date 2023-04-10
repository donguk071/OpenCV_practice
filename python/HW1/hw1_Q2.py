from math import nan
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("./HW1/contrast_test/lowContrastCam.jpg")
img2 = cv.imread("./HW1/contrast_test/defaultCam.jpg")

img = cv.resize(img, dsize=(768, 1024))
img2 = cv.resize(img2, dsize=(768, 1024))
cv.imwrite(filename="lowContrast.jpg", img=img)
cv.imwrite(filename="default.jpg", img=img2)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_origin = img.copy()
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

print(img.shape)

# implement


def colorHE1(img):
    imgRGB_HE = []
    imgRGB = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]

    for i in range(3):
        hist, bins = np.histogram(imgRGB[i].flatten(), 256, [0, 256])
        cdf = np.cumsum(hist)
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max() - cdf_m.min())
        cdf_result = np.ma.filled(cdf_m, 0).astype('uint8')
        imgRGB_HE.append(cdf_result[imgRGB[i]])

    img_HE = cv.merge((imgRGB_HE[0], imgRGB_HE[1], imgRGB_HE[2]))
    return img_HE

# using HE method


def colorHE2(img):
    imgR = cv.equalizeHist(img[:, :, 0])
    imgG = cv.equalizeHist(img[:, :, 1])
    imgB = cv.equalizeHist(img[:, :, 2])
    img_HE = cv.merge((imgR, imgG, imgB))
    return img_HE


# using CLAHE method


def colorAHE(img):
    imgR = cv.createCLAHE(
        clipLimit=nan, tileGridSize=(8, 8)).apply(img[:, :, 0])
    imgG = cv.createCLAHE(
        clipLimit=nan, tileGridSize=(8, 8)).apply(img[:, :, 1])
    imgB = cv.createCLAHE(
        clipLimit=nan, tileGridSize=(8, 8)).apply(img[:, :, 2])
    img_AHE = cv.merge((imgR, imgG, imgB))
    return img_AHE


# using CLAHE method


def colorCLAHE(img):
    imgR = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(img[:, :, 0])
    imgG = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(img[:, :, 1])
    imgB = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(img[:, :, 2])
    img_CLAHE = cv.merge((imgR, imgG, imgB))
    return img_CLAHE


def enhanceHE(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img


img_HE1 = colorHE1(img)
img_HE2 = colorHE2(img)
img_AHE = colorAHE(img)
img_CLAHE = colorCLAHE(img)

img_enhanceHE = enhanceHE(img)

titles = ["low contrast", "HE", "AHE", "CLAHE"]
images = [img_origin, img_HE1, img_AHE, img_CLAHE]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

cv.imshow("enhanceHE", img_enhanceHE)
plt.show()
