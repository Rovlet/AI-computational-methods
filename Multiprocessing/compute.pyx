import cython
cimport numpy as np

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve(float[:, :, ::1] image, long[:, ::1] kernel, int padding=1):
    cdef int image_height, image_width, channel_count
    cdef int kernel_height, kernel_width, kernel_halfh, kernel_halfw
    cdef int x_min, x_max, y_min, y_max, x, y, u, v, c
    cdef float value, tmp, total


    #image = np.pad(image, ((0,padding, padding), (0,padding, padding), (0,padding, padding) ))

    #image=cv2.copyMakeBorder(image,padding,padding,padding,padding,int)
    image = np.pad(image, pad_width=[(0, 0), (padding, padding), (padding, padding)])
    channel_count = image.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]

    kernel_height = kernel.shape[0]
    kernel_halfh = kernel_height // 2
    kernel_width = kernel.shape[1]
    kernel_halfw = kernel_width // 2

    cdef np.ndarray output = np.zeros([channel_count, image_height, image_width], dtype=np.int32)

    # Do convolution
    for x in range(image_width):
        for y in range(image_height):
            # Calculate usable image / kernel range
            x_min = max(0, x - kernel_halfw)#>=0
            x_max = min(image_width - 1, x + kernel_halfw)
            y_min = max(0, y - kernel_halfh)#>=0
            y_max = min(image_height - 1, y + kernel_halfh)
            # Convolve filter
            for c in range(channel_count):
                value = 0
                for u in range(x_min, x_max + 1):
                    for v in range(y_min, y_max + 1):
                        tmp = kernel[v - y + kernel_halfh, u - x + kernel_halfw]
                        value += image[c, v, u] * tmp
                if value * 255 >= 70:
                    output[c, y, x] = 255
                elif value * 255 < 70:
                    output[c, y, x] = 0
    output = output[:, 1:-1, 1:-1]
    return output
