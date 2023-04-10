import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def combineImage(img, img2, a, deduplication=True):

    h, w, c = img.shape
    h2, w2, c = img2.shape

    paddingT = int((h - h2 + 1)*2 / 3) + a
    paddingB = int((h - h2) / 3) - a
    paddingL = int((w - w2) / 2)
    paddingR = int((w - w2) / 2)

    img2 = cv.copyMakeBorder(img2, paddingT, paddingB,
                             paddingL, paddingR, cv.BORDER_CONSTANT, (0, 0, 0))

    if deduplication:
        img_sub = img[paddingT: - paddingB, paddingL: -paddingR, :]
        img_sub = cv.copyMakeBorder(img_sub, paddingT, paddingB,
                                    paddingL, paddingR, cv.BORDER_CONSTANT, (0, 0, 0))

        img_sub = cv.absdiff(img, img_sub)
        img_combined = cv.add(img_sub, img2)
    else:
        img_combined = cv.add(img, img2)
    return img_combined


def generate_gaussian_pyramid(img, levels):
    GP = [img]
    for i in range(1, levels):  # 1 to levels - 1 same as range(1, levels, 1)
        img = cv.pyrDown(img)
        GP.append(img)
    return GP


def __pyrUp(img, size=None):
    nt = tuple([2*x for x in img.shape[:2]])
    print("target nt size : ", nt)
    if size == None:
        size = nt
    # bug?!
    if nt != size:
        upscale_img = cv.pyrUp(img, None, nt)
    else:
        upscale_img = cv.pyrUp(img)
    return upscale_img


def generate_laplacian_pyramid(GP):
    levels = len(GP)
    LP = []  # [GP[levels + 1]] # 마지막은 없으니까
    for i in range(levels - 1, 0, -1):
        # upsample_img = cv.pyrUp(GP[i])
        print("laplacian in : ", GP[i].shape)
        upsample_img = __pyrUp(GP[i], (GP[i-1].shape[0], GP[i-1].shape[1]))
        print("laplacian out : ", upsample_img.shape)

        laplacian_img = cv.subtract(GP[i-1], upsample_img)
        LP.append(laplacian_img)
    LP.reverse()
    return LP


def generate_pyramid_composition_image(Pimgs):
    levels = len(Pimgs)
    # print(levels)
    rows, cols = Pimgs[0].shape[:2]
    composite_image = np.zeros(
        (rows, cols + int(cols / 2 + 0.5), 3), dtype=Pimgs[0].dtype)
    composite_image[:rows, :cols, :] = Pimgs[0]
    i_row = 0
    for p in Pimgs[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image


def laplacian_combine_pyramid(LP_hand, LP_eye, padding, plus=1000):
    laplacian_combine = []
    for i in range(len(LP_hand)):
        if i == plus:
            laplacian_combine.append(combineImage(
                LP_hand[i], 2*LP_eye[i], int(padding/(2**i)), False))
        else:
            laplacian_combine.append(combineImage(
                LP_hand[i], LP_eye[i], int(padding/(2**i)), False))
    return laplacian_combine


def colorCLAHE(img):
    imgR = cv.createCLAHE(
        clipLimit=1, tileGridSize=(8, 8)).apply(img[:, :, 0])
    imgG = cv.createCLAHE(
        clipLimit=1, tileGridSize=(8, 8)).apply(img[:, :, 1])
    imgB = cv.createCLAHE(
        clipLimit=1, tileGridSize=(8, 8)).apply(img[:, :, 2])
    img_CLAHE = cv.merge((imgR, imgG, imgB))
    return img_CLAHE


# starting point
img = cv.imread("./HW2/hand.jpg")
img2 = cv.imread("./HW2/eye3.jpg")

img = cv.resize(img, dsize=(480, 640))
img2 = cv.resize(img2, dsize=(128, 64))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img_combined = combineImage(img, img2, 16)  # combine two img

GP_hand = generate_gaussian_pyramid(img, 6)
GP_eye = generate_gaussian_pyramid(img2, 6)
Gp_combined = generate_gaussian_pyramid(img_combined, 6)
LP_hand = generate_laplacian_pyramid(GP_hand)
LP_eye = generate_laplacian_pyramid(GP_eye)

# laplacian_comdine = []
# laplacian_comdine.append(combineImage(LP_hand[0], 2*LP_eye[0], 16, False))
# laplacian_comdine.append(combineImage(LP_hand[1], LP_eye[1], 8, False))
# laplacian_comdine.append(combineImage(LP_hand[2], LP_eye[2], 4, False))
# laplacian_comdine.append(combineImage(LP_hand[3], LP_eye[3], 2, False))
# laplacian_comdine.append(combineImage(LP_hand[4], LP_eye[4], 1, False))

laplacian_comdine = laplacian_combine_pyramid(LP_hand, LP_eye, 16, 0)

composition_image = []
composition_image.append(generate_pyramid_composition_image(Gp_combined))
composition_image.append(3*generate_pyramid_composition_image(LP_hand))
composition_image.append(3*generate_pyramid_composition_image(LP_eye))
composition_image.append(
    3*generate_pyramid_composition_image(laplacian_comdine))

for i in composition_image:
    i = cv.cvtColor(i, cv.COLOR_RGB2BGR)
    print(i.shape)
    cv.imshow("title", i)
    cv.waitKey(0)


final_img = []

# final_img.append(cv.pyrUp(Gp_combined[3]) + laplacian_comdine[2])
# final_img.append(cv.pyrUp(final_img[0]) + laplacian_comdine[1])
# final_img.append(cv.pyrUp(final_img[1]) + laplacian_comdine[0])
# final_img.append(cv.pyrUp(final_img[1]) + laplacian_comdine[0])

final_img.append(cv.pyrUp(Gp_combined[4]) + LP_hand[3])
final_img.append(cv.pyrUp(final_img[0]) + LP_hand[2])
final_img.append(cv.pyrUp(final_img[1]) + laplacian_comdine[1])
final_img.append(cv.pyrUp(final_img[2]) + laplacian_comdine[0])


titles = ["1", "2", "3", "4", "original"]
images = [final_img[0], final_img[1], final_img[2],
          final_img[3], img_combined]

for i in range(len(titles)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(final_img[3])
plt.show()

final_img = colorCLAHE(final_img[3])
plt.imshow(final_img)
plt.show()

plt.imshow(img_combined)
plt.show()
