
import numpy as np
import cv2
from numba import cuda



def random_image(size):
    return np.random.randint(0, 2, size=size).astype(float)

@cuda.jit(device=True)
def lerp(a, b, t):
    return a * (1 - t) + b * t

@cuda.jit()
def cuda_kernel_blur(img, new_img, factor, radius):
    i, j = cuda.grid(2)
    if radius == 0 or factor == 0:
        new_img[i, j] = img[i, j]
        return
    s = 0
    k = 0
    for u in range(-radius, radius+1):
        _i = i + u
        if _i >= img.shape[0]:
            break
        if _i < 0:
            continue
        for v in range(-radius, radius+1):
            _j = j + v
            if _j >= img.shape[1]:
                break
            if _j < 0:
                continue
            if u == 0 and v == 0 or u*u + v*v > radius:
                continue
            s += img[_i, _j]
            k += 1
    t = lerp(1 / (k + 1), 1, 1 - factor)
    new_img[i, j] = s * (1 - t) / k + img[i, j] * t


def blur(img, factor=0.5, radius=1, iterations=1):
    _img = cuda.to_device(img)
    _new_img = cuda.device_array_like(_img)
    for _ in range(iterations):
        cuda_kernel_blur[(img.shape[0] // 32, img.shape[1] // 32),
                         (32, 32)](_img, _new_img, factor, radius)
        _img.copy_to_device(_new_img)
    new_img = _new_img.copy_to_host()
    return new_img

def threshold(img, t):
    return np.where(img < t, False, True)

def random_mask(size, weight = 0.5):
    img = random_image(size)
    img = blur(img, factor=1, radius=4, iterations=min(size[0], size[1])//2)
    img = threshold(img, weight)
    return img

def hybridize_images(img0, img1, mask):
    return np.where(mask, img0, img1)



if __name__ == '__main__':
    pimg = random_mask((1080, 1920))
    while True:
        cimg = random_mask((1080, 1920))
        mask = random_mask((1080, 1920))
        img = hybridize_images(pimg, cimg, mask)
        cv2.imshow('test', img.astype(float))
        k = cv2.waitKey() & 0xFF
        if k == 27: #ESCAPE
            break
        pimg = cimg
