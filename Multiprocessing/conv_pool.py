import cv2
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing


def get_output(arguments):
    y_filter_shape, y_img_shape, x_img_shape, x_filter_shape, kernel, img_padded, y = arguments
    output = []
    if y > y_img_shape - y_filter_shape:
        return
    for x in range(x_img_shape):
        if x > x_img_shape - x_filter_shape:
            break
        output.append((kernel * img_padded[x:x + x_filter_shape, y: y + y_filter_shape]).sum())
    return output


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

    items = [[y_filter_shape, y_img_shape, x_img_shape,  x_filter_shape, kernel, img_padded, v] for v in range(y_img_shape)]

    i = 0
    with multiprocessing.Pool(4) as pool:
        results = pool.map(get_output, items)
    for i in range(len(results)):
        if results[i]:
            output[:, i] = results[i]
    return output


if __name__ == '__main__':
    image = cv2.imread(r'lenna.png')
    filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    output = conv(image, filter)
    plt.imshow(output)
    cv2.imwrite('R1.jpg', output)
