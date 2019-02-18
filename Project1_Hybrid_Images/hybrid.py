import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # get dimensions of kernel
    k_width, k_height = kernel.shape
    # calculate the amount of zeros to pad the image with
    pad_width = (k_width - 1) / 2
    pad_height = (k_height - 1) / 2
    # create a new zero filled array with the same dimensions as the image
    output = np.zeros(img.shape)
    if img.ndim == 2:
        # two dimensional image - grayscale - get the dimensions
        i_width, i_height = img.shape
        # pad the image with zeros, saving output to a new copy
        img_copy = np.pad(img, [(pad_width, pad_width), (pad_height, pad_height)], 'constant', constant_values=((0,0), (0,0)))

        # loop through the original image, not the padded one
        for i in range(i_width):
            for j in range(i_height):
                # get the range of pixels to multiply with from the padded image
                pixel_view = img_copy[i:i+k_width, j:j+k_height]
                # mult the two matrixes and sum the values, then save in the output array
                output[i,j] = np.sum(pixel_view * kernel)
    else:
        # three dims - rgb image - get the dimensions
        i_width, i_height, channels = img.shape
        # loop through the channels
        for channel in range(channels):
            # create a view of the current channel
            img_channel = img[:, :, channel]
            # pad the current channel and copy it
            img_copy = np.pad(img_channel, [(pad_width, pad_width), (pad_height, pad_height)], 'constant', constant_values=((0,0), (0,0)))

            # loop through the original channel, not the padded copy
            for i in range(i_width):
                for j in range(i_height):
                    # get the range of pixels to multiply with from the padded image
                    pixel_view = img_copy[i:i+k_width, j:j+k_height]
                    # mult the two matrixes and sum the values, then save in the output array
                    output[i, j, channel] = np.sum(pixel_view * kernel)
    return output


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # just flip the kernel on both axes
    flipped = np.flip(kernel, (0, 1))
    # and return the cross correlation of the image with the flipped kernel
    return cross_correlation_2d(img, flipped)

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    # create a new zero filled array of the given dimensions
    output = np.zeros((height, width))
    # calculate the coordinates of the central cell
    center_x = int(height / 2)
    center_y = int(width / 2)
    # calculate the constant of the gaussian function
    constant = 1. / (2. * np.pi * (sigma ** 2))
    # ditto for the denominator
    exponent_denominator = 2. * (sigma ** 2)
    # for each cell in the output array...
    for x in range(height):
        for y in range(width):
            # calculate the numerator using the relative location of the cell from the center
            exponent_numerator = ((x - center_x) ** 2) + ((y - center_y) ** 2)
            exponent = (-1.) * exponent_numerator / exponent_denominator
            # save it in the output array
            output[x, y] = constant * np.power(np.e, exponent)
    # normalize it
    return output / np.sum(output)

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # just get a gaussian kernel and convolve
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # subtract the convolved from the original image to keep the high frequency values
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    convolved = convolve_2d(img, kernel)
    return img - convolved

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

