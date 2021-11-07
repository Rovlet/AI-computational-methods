import cv2
from matplotlib import pyplot as plt
import numpy as np
from compute import convolve


def conv(img, filter, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(filter))
    x_filter_shape = kernel.shape[0]
    y_filter_shape = kernel.shape[1]
    x_img_shape = img.shape[0]
    y_img_shape = img.shape[1]

    x_output = int(((x_img_shape - x_filter_shape + 2*padding) / strides) + 1)
    y_output = int(((y_img_shape - y_filter_shape + 2*padding) / strides) + 1)
    output = np.zeros((x_output, y_output))
    img_padded = img
    for y in range(y_img_shape):
        if y > y_img_shape - y_filter_shape:
            break
        for x in range(x_img_shape):
            if x > x_img_shape - x_filter_shape:
                break
            output[x, y] = (kernel * img_padded[x:x + x_filter_shape, y: y + y_filter_shape]).sum()
    return output



if __name__ == '__main__':
    image = cv2.imread(r'lenna.png')
    kernel = np.flipud(np.fliplr(filter))
    padding = 0
    strides = 1
    x_filter_shape = kernel.shape[0]
    y_filter_shape = kernel.shape[1]
    x_img_shape = image.shape[0]
    y_img_shape = image.shape[1]
    x_output = int(((x_img_shape - x_filter_shape + 2 * padding) / strides) + 1)
    y_output = int(((y_img_shape - y_filter_shape + 2 * padding) / strides) + 1)
    output = np.zeros((x_output, y_output))
    filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    output = convolve(image, output, filter)
    plt.imshow(output)
    cv2.imwrite('R1.jpg', output)



