import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def crop_image(src, top, bottom, left, right):
    img = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    crop_img = img[top: bottom, left: right, :]
    return crop_img


img = cv.imread("./HW2/eyeh.jpg")
img = crop_image(img, 350, 400, 200, 350)

plt.imshow(img)
plt.show()
cv.imwrite(filename="eye.jpg", img=img)
