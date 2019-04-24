# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # G= np.empty([9,1])
    # for i,image in enumerate(images):
    #     np.insert(G,lights[i] * image, axis=1)
        
    # images = np.array([[[coord for coord in xk] for xk in xj] for xj in xi], ndmin=3)
    # images = np.array(images).reshape(3,9)
    # g1=np.linalg.inv(np.matmul(np.transpose(lights),  lights))
    # g2= np.transpose(lights) * images

    # G = np.matmul(g1,g2)
    # print(G.shape())

    # albedo = np.linalg.norm(G)
    # normals = G/albedo

    # return albedo, normals

    base_image = images[0]
    n = len(images)
    image_shape = base_image.shape
    h, w, c = image_shape

    I = np.array(images).reshape(n, h * w * c)
    l_inv = np.linalg.inv(np.dot(lights.T, lights))

    l_t_l = np.dot(l_inv, lights.T)

    G = np.dot(l_t_l, I)

    G_channels = np.reshape(G.T,(h, w, c, 3))
    albedos = np.linalg.norm(G_channels, axis = 3)

    G_grey = np.mean(G_channels, axis=2)
    norm_of_albedos = np.linalg.norm(G_grey, axis = 2)

    threshold = 1e-7
    normals = G_grey/np.maximum(threshold, norm_of_albedos[:,:,np.newaxis])
    normals[norm_of_albedos < threshold] = 0
    
    return albedos, normals

    #taking array of images with corresponding light directions, and computing map of combined images
    # two different ways, albedo way and normal way
    # red is right direction (+x)
    # green is left direction (+y)
    # blue is normal pointing out of screen (+z)
    #kd is norm of G(square root of sum)
    # combining image and lighting arrays
    #light x image array which gets you G and then decompose to albedo of normals
    # inverse (L transpose * L) 
    # multiply ^ by ( l transpose times images)
    # that's G
    # take np.linalg.norm(G) => albedos
    # divide G by ^ to get normals
    # 

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    #raise NotImplementedError()

    #k times rt times identity 
    P = np.dot(K, Rt)
    h, w, _ = np.shape(points)
    ones = np.ones((h, w, 1))
    points_homography = np.concatenate((points, ones), axis=2)

    projections = np.tensordot(points_homography, P.T, axes = 1)
    normalized = projections / (projections[:,:,2])[:,:,np.newaxis]
    return normalized[:,:,0:2]

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    #raise NotImplementedError()

    h, w, c = image.shape
    normalized_shape = (h, w, c * ncc_size**2)
    normalized = np.zeros(normalized_shape, dtype = np.float32)

    mid = ncc_size / 2

    for i in range(mid, h - mid):
        for j in range(mid, w - mid):
            all_windows = []
            for k in range(c):
                window = image[i - mid : i + mid + 1, j - mid : j + mid + 1, k]
                window = (window - np.mean(window)).flatten()
                all_windows.append(window.T)
            flattened = np.array(all_windows).flatten(order='C')
            norm = np.linalg.norm(flattened)
            if norm < 1e-6:
                flattened.fill(0)
            else:
                flattened = flattened / norm
            normalized[i,j] = flattened
    return normalized

def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (c * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(np.multiply(image1, image2), axis = 2)
    return ncc