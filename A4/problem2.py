from ctypes.wintypes import HACCEL
import numpy as np


def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calculated SSD cost as a floating point value
    """

    #
    # Your code goes here
    #
    cost_ssd = 0

    m = patch1.shape[0] #We assume path1.shape == path2.shape

    for x in range(m):
        for y in range(m):
            cost_ssd += pow(patch1[x][y] - patch2[x][y], 2)


    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calculated NC cost as a floating point value
    """

    #
    # Your code goes here
    #
    cost_nc = 0

    patch1Mean = np.mean(patch1)
    patch2Mean = np.mean(patch2)
    m = patch1.shape[0] #We assume path1.shape == path2.shape
    wL = np.reshape(patch1, (m**2, 1))
    wR = np.reshape(patch2, (m**2, 1))

    normLeft = np.sum(np.abs(wL - patch1Mean))
    normRight = np.sum(np.abs(wR - patch2Mean))

    topResult = np.dot(np.transpose(wL - patch1Mean),(wR - patch2Mean))[0][0] #We need the [0][0] here due to transpose() which changes organization inside the matrix
    bottomResult = normLeft * normRight
    cost_nc = topResult / bottomResult

    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value        
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """

    assert patch1.shape == patch2.shape 

    #
    # Your code goes here
    #
    m = patch1.shape[0]
    cost_val = (1/m**2)*cost_ssd(patch1, patch2) + alpha*cost_nc(patch1, patch2)

    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    #
    # Your code goes here
    #
    dimensions_img = input_img.shape #For debug purpose
    padded_img = input_img.copy()

    padded_img = np.pad(padded_img, window_size * 2, padding_mode)

    #To check shape of result
    #print("Dim before : ", dimensions_img)
    #print("Dim after : ", padded_img.shape)


    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2 
    assert padded_img_r.ndim == 2 
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    disparity = []

    #I create these index to start where the original image is supposed to start (without padding)
    leftStartIndexX = (2*window_size) - (window_size - 1)//2 #According to this axis : img[x][y]
    leftEndIndexX = (2*window_size) + (window_size - 1)//2
    leftStartIndexY = (2*window_size) - (window_size - 1)//2
    leftEndIndexY = (2*window_size) + (window_size - 1)//2

    rightStartIndexX = (2*window_size) - (window_size - 1)//2
    rightEndIndexX = (2*window_size) + (window_size - 1)//2

    dimensions = padded_img_l.shape
    originalH = dimensions[0] - 4*window_size #To get the original I subtract 2*window_size according to the padding I did, two times (left and right/top and bottom)
    originalW = dimensions[1] - 4*window_size

    #Used to display percentage progression see line 183
    nbrPixel = originalW * originalH
    nbrIteration = nbrPixel * max_disp
    iterationNumero = 0
    pourcentage = 0 

    #First "real" pixel is at [window_size - 1][window_size - 1] according to our padding (padding for window_size exactly is useless even if it's not that expensive in computation I guess)
    for x in range(originalH):
        for y in range(originalW):
            xyCost = False
            dPixel = False
            for d in range(max_disp):
                
                rightStartIndexY = (2*window_size) - (window_size - 1)//2 - d
                rightEndIndexY = (2*window_size) + (window_size - 1)//2 - d
                
                patchLeft = padded_img_l[x + leftStartIndexX : x + leftEndIndexX + 1, y + leftStartIndexY : y + leftEndIndexY + 1] #+ 1 because last index is excluded 
                patchRight = padded_img_r[x + rightStartIndexX : x + rightEndIndexX + 1,y + rightStartIndexY : y + rightEndIndexY + 1]

                cost = cost_function(patchLeft,patchRight,alpha)

                if(xyCost == False): #First iteration
                    xyCost = cost
                    dPixel = d

                if(cost < xyCost):
                    xyCost = cost
                    dPixel = d

                #Uncomment to check % progression
                iterationNumero += 1 
                if((iterationNumero/nbrIteration)*100 >= pourcentage + 1):
                    pourcentage += 1
                    print(pourcentage, "%...")
            disparity.append(dPixel)


    

    disparity = np.asarray(disparity)
    disparity = np.reshape(disparity,(originalH,originalW))

    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """

    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape
    H = disparity_gt.shape[0]
    W = disparity_gt.shape[1]
    N = H * W

    intermediateMatrix = np.abs(disparity_gt - disparity_res)

    aepe = (1/N) * np.sum(intermediateMatrix)

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #
    alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    alpha = -0.01
    #Tests with max_disp = 15 and window_size = 11
    #alpha -> AEPE
    #-0.06 -> 0.922
    #-0.01 -> 0.913
    #0.04 -> 1.468
    #0.1 -> 1.823
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """
        #According to the lecture and slide "Influence of window_size, Q1 -> 0 and Q2 -> 1 (starting from 0-indexing)"
        return (1, 2, 2, 3)
