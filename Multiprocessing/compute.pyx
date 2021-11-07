# cython: language_level=3
# cython: infer_types=True
cimport cython
cimport numpy as np

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve(np.ndarray[:, :] image, np.ndarray[:, :] output, np.ndarray[:, :] kernel):
    cdef int image_height, image_width, channel_count
    cdef int kernel_height, kernel_width
    cdef int x_min, x_max, y_min, y_max, x, y, u, v, c

    cdef float value, tmp, total

    image_height = image.shape[0]
    image_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_halfh = kernel_height//2
    kernel_width = kernel.shape[1]
    kernel_halfw = kernel_width//2

    # Do convolution
    for y in range(image_height):
        if y > image_height - kernel_height:
            break
        for x in range(image_width):
            if x > image_width - kernel_width:
                break
            x_min = max(0, x - kernel_halfw)
            x_max = min(image_width - 1, x + kernel_halfw)
            y_min = max(0, y - kernel_halfh)
            y_max = min(image_height - 1, y + kernel_halfh)
            for u in range(x_min, x_max + 1):
                for v in range(y_min, y_max + 1):
                    tmp = kernel[v - x + kernel_halfh, u - y + kernel_halfw]
                    value += image[v, u] * tmp
                    total += tmp
            output[x, y] = value / total
            # output[x, y] = (kernel * image[x:x + kernel_width, y: y + kernel_height]).sum()
    return output

    # for x in range(image_width):
    #     for y in range(image_height):
    #         # Calculate usable image / kernel range
    #         x_min = max(0, x - kernel_halfw)
    #         x_max = min(image_width - 1, x + kernel_halfw)
    #         y_min = max(0, y - kernel_halfh)
    #         y_max = min(image_height - 1, y + kernel_halfh)
    #
    #         # Convolve filter
    #         for c in range(channel_count):
    #             value = 0
    #             total = 0
    #             for u in range(x_min, x_max + 1):
    #                 for v in range(y_min, y_max + 1):
    #                     tmp = kernel[v - y + kernel_halfh, u - x + kernel_halfw]
    #                     value += image[c, v, u] * tmp
    #                     total += tmp
    #             output[c, y, x] = value / total