from cv2 import Sobel
import pywt
import cv2
import numpy as np


def get_wavelets(im, wavelet='haar'):
    return pywt.dwt2(im, wavelet)


def normalize_image(im, alpha=0, beta=255, dtype='int8'):
    if dtype == 'int8':
        dtype_cv2 = cv2.CV_8S
    elif dtype == 'float32':
        dtype_cv2 = cv2.CV_32F
    return cv2.normalize(
        im, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=dtype_cv2)


def concatenate_channels(splitted_image, transform):
    if transform == 'wavelets':
        LL, (LH, HL, HH) = splitted_image
        im = cv2.merge([LH, HL, HH])
        im = normalize_image(im, alpha=0, beta=255, dtype='float32')
        return im
    elif transform == 'canny_sobel':
        im = cv2.merge(splitted_image)
        im = normalize_image(im, alpha=0, beta=255, dtype='float32')
        return im

    else:
        raise Exception("Just wavelets function implemented")


def apply_canny(im, th1, th2):
    return cv2.Canny(im, th1, th2)


def apply_sobel(im, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=3):
    return cv2.Sobel(im, ddepth=ddepth, dx=dx, dy=dy, ksize=ksize)


def process_transform(im: np.ndarray, transform: str, **kwargs):
    if transform == 'wavelets':
        if len(im.shape) > 2:
            im = im[:, :, 0]
        im_wavelets = get_wavelets(im)
        im_transformed = concatenate_channels(im_wavelets, transform)
    elif transform == 'canny_sobel':
        blur_image = cv2.GaussianBlur(im, (3, 3), 0)
        canny = apply_canny(blur_image, th1=75, th2=150)
        sobelxy = apply_sobel(blur_image)
        norm_sobelxy = normalize_image(sobelxy, dtype='int8')
        norm_sobelxy = norm_sobelxy.astype(np.uint8)
        im_transformed = concatenate_channels([im, canny, norm_sobelxy], transform)
    return im_transformed
