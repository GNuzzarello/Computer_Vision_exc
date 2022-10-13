import numpy as np
from numpy.lib.type_check import imag
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """
    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    bayerdataDim = bayerdata.shape

    red = np.zeros(shape=(bayerdataDim[0],bayerdataDim[1]))
    green = np.zeros(shape=(bayerdataDim[0],bayerdataDim[1]))
    blue = np.zeros(shape=(bayerdataDim[0],bayerdataDim[1]))

    #Our bayer pattern has the following form :
    #GRGRGRGR
    #BGBGBGBG
    #And so on....

    #Starting from now, we will work line by line, each line is either an Green Red one (GR Line) or a Blue Green one (BG Line)
    #We will attribute to each value in bayerdata one color according to our bayer pattern described above

    lineParity = False # False -> GR Line True -> BG Line 
    rowParity = False #To select one of the two colour of the line

    for x in range(bayerdataDim[0]): #As mentionned, we work line by line and we will give one colour for each value in bayerdata
        rowParity = False #To be aware with wich one I'm starting
        for y in range(bayerdataDim[1]):
            if(not(lineParity)):
                if(not(rowParity)):
                    green[x][y] = bayerdata[x][y]
                else:
                    red[x][y] = bayerdata[x][y]
            else:
                if(not(rowParity)):
                    blue[x][y] = bayerdata[x][y]
                else:
                    green[x][y] = bayerdata[x][y]
            rowParity = not(rowParity) #We go to the next color of the line
        lineParity = not(lineParity) #We go to the next line (GR or BG)

    return red, green, blue


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    #Assuming that r,g and b have same dimensions 
    dimensions = r.shape
    imageArray = np.zeros(shape=(dimensions[0],dimensions[1],3))
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            imageArray[x][y][0] = r[x][y]
            imageArray[x][y][1] = g[x][y]
            imageArray[x][y][2] = b[x][y]

    return imageArray


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #We will use kernel given by http://www.sfu.ca/~gchapman/e895/e895l11.pdf (Simon Fraser University)
    #Here these kernels are working because when we apply convolution to our pattern (described in separatechannels()) it selects only corresponding colour
    #Per example for the red one we have :
    
    # G R G R G R
    # B G B G B G
    # G R G R G R

    #However here G and B equals 0 (because we are in R field)
    #So when we apply the first operation of the convolution with RB kernel, we will "have" 4*R, which will be divided by 4 with the scalar.
    #Then we slide one time to the right side, and the result is the same 
    #This is the same logic for G filter, which uses a different kernel because as we saw in lecture, green twices number of red and blue value


    kernelRB = np.array([[1,2,1],[2,4,2],[1,2,1]])
    kernelRB = np.true_divide(kernelRB,4)
    kernelG = np.array([[0,1,0],[1,4,1],[0,1,0]])
    kernelG = np.true_divide(kernelG,4)

    rInterpolated = convolve(r, kernelRB, mode='nearest') #cvalue -> 0 (default)
    gInterpolated = convolve(g, kernelG, mode='nearest') #cvalue -> 0 (default)
    bInterpolated = convolve(b, kernelRB, mode='nearest') #cvalue -> 0 (default)


    imageArray = assembleimage(rInterpolated,gInterpolated,bInterpolated)

    return imageArray
