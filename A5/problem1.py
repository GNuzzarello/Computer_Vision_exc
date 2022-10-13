import numpy as np
from scipy.ndimage import convolve
from functools import partial
from scipy import interpolate
from PIL import Image



######################
# Basic Lucas-Kanade #
######################

conv2d = partial(convolve, mode="mirror")


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape
    
    Ix = np.empty_like(im1)
    Iy = np.empty_like(im1)
    It = np.empty_like(im1)

    Dx = 1/2 * np.array([[1,0,-1]])
    Dy = Dx.T

    Ix = conv2d(im1, Dx)
    Iy = conv2d(im1, Dy)

    It = im2 - im1
    
    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    (h, w) = Ix.shape[0], Ix.shape[1]
    Ixx = conv2d(Ix**2, np.ones((patch_size, patch_size)))
    Iyy = conv2d(Iy**2, np.ones((patch_size, patch_size)))
    Ixy = conv2d(Iy*Ix, np.ones((patch_size, patch_size)))
    Ixt = conv2d(Ix*It, np.ones((patch_size, patch_size)))
    Iyt = conv2d(Iy*It, np.ones((patch_size, patch_size)))

    st = np.zeros((2,2))
    rhs = np.zeros((2,1))
    
    for i in range(0, h):
        for j in range(0, w):
            st = np.array([[Ixx[i,j], Ixy[i,j]], [Ixy[i,j], Iyy[i,j]]])
            rhs = - np.array([[Ixt[i,j]], [Iyt[i,j]]])
            rank = np.linalg.matrix_rank(st)

            if rank != 2:
                inv = np.zeros((2,2))
            else:
                inv = np.linalg.inv(st)
            
            (u[i,j], v[i,j]) = inv @ rhs
        
    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape
    
    im_warp = np.empty_like(im)
    X, Y = np.meshgrid(np.arange(0,im.shape[1],1), np.arange(0,im.shape[0],1))
    pu, pv = np.ndarray.flatten(X+u), np.ndarray.flatten(Y+v)

    im_warp = interpolate.griddata((pu, pv), np.ndarray.flatten(im), (X, Y), method='linear', fill_value=0.0)

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0
    d = np.sum((im1-im2)**2)

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2
    Hint: Don't forget to smooth the image beforhand (in this function).

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """


    g_k = gaussian_kernel(fsize, sigma)
    x = conv2d(x, g_k)
    x = x[::2, ::2]

    return x

def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is a sequence of downscaled images
    (here, by a factor of 2 w.r.t. the previous image in the pyramid)

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images in ascending order of resolution
    '''
    
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample_x2(pyramid[i], fsize, sigma))

    return [img]

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape
    
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    n_levels = len(pyramid1)


    for level in range(0, n_levels):
        im2 = pyramid2[n_levels-level-1]
        print("level: ", level)
        for i in range(0, n_iter):
            print("   iteration: ", i)
            im1 = warp(pyramid1[n_levels-level-1], u, v)
            Ix, Iy, It = compute_derivatives(im1, im2) # gradients
            uu, vv = compute_motion(Ix, Iy, It) # flow
            u += uu
            v += vv
        if level != n_levels-1:
            u = np.array(Image.fromarray(u * 2).resize(pyramid1[n_levels-level-2].shape[::-1]))
            v = np.array(Image.fromarray(v * 2).resize(pyramid1[n_levels-level-2].shape[::-1]))
                
    assert u.shape == im1.shape and \
            v.shape == im1.shape
    return u, v