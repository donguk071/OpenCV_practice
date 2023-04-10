import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def localOstu(img, blockSize, c):
    img_out = img
    if blockSize % 2 == 0:
        print("local의 크기는 홀수만 입력이 가능합니다")
        return img
    paddingSize = int(blockSize/2)
    img = cv.copyMakeBorder(img, paddingSize, paddingSize,
                            paddingSize, paddingSize, cv.BORDER_REFLECT)
    img_out = cv.copyMakeBorder(img_out, paddingSize, paddingSize,
                                paddingSize, paddingSize, cv.BORDER_REFLECT)

    for i in range(len(img) - paddingSize):
        for j in range(len(img[i]) - paddingSize):
            tempimg = img[i:i+blockSize, j:j+blockSize]
            ret, dummy = cv.threshold(
                tempimg, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
            #print("크기 : ", tempimg.shape, " ", ret)
            if img_out[i+paddingSize][j+paddingSize] > ret - c:
                img_out[i+paddingSize][j+paddingSize] = 255
            else:
                img_out[i+paddingSize][j+paddingSize] = 0
    img_out = img_out[paddingSize:-paddingSize, paddingSize:-paddingSize]
    return img_out


img = cv.imread("./HW1/test1.jpg", 0)
img = cv.medianBlur(img, 3)
#img = cv.GaussianBlur(img, (3, 3), 0)
localSize = 3

# img = cv.resize(img, dsize=(512, 512))
# cv.imwrite(filename="test1.jpg", img=img)

print(img.shape)
print(len(img))
print(len(img[1]))

img2 = localOstu(img, localSize, 1)

print(img2.shape)
print(len(img2))
print(len(img2[1]))

cv.imshow('origin', img)
cv.imshow('result', img2)
cv.waitKey(0)
