import numpy as np
import matplotlib.pyplot as plt
import numpy as npy

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    plt.imshow(img)
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """
    npy.save(path,img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """
    return npy.load(path)


def mirror_horizontal(img): #https://numpy.org/doc/stable/reference/generated/numpy.flip.html
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    return npy.flip(img, axis=1) #fliplr equivalent


def display_images(img1, img2): #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig = plt.suptitle("Display images")

    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

    
