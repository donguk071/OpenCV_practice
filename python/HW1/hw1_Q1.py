import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("./HW1/potato.png", 0)
#img = cv.medianBlur(img, 3)
# img = cv.medianBlur(img, 3)

# img = cv.medianBlur(img, 3)
# img = cv.medianBlur(img, 3)

# img = cv.medianBlur(img, 9)

# img = cv.medianBlur(img, 3)


# img = cv.medianBlur(img, 3)
# img = cv.medianBlur(img, 3)


img = cv.resize(img, dsize=(600, 600), interpolation=cv.INTER_AREA)

th_v, th1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 599, 5)  # 평균에서 차감할값
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 299, 5)
th4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 109, 5)
th5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 49, 5)
th6 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 29, 5)
th7 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 9, 5)
th8 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 3, 5)

titles = ["original", "global", "599", "299", "109", "49", "29", "9", "3"]
images = [img, th1, th2, th3, th4, th5, th6, th7, th8]

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
