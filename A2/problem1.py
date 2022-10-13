import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import convolve


def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    imgs = []
    feats = []

    pathImage = path + "/facial_images/"
    fileList = os.listdir(pathImage)

    for file in fileList:
        with open(pathImage + file, 'rb') as pgmf: #https://stackoverflow.com/questions/35723865/read-a-pgm-file-in-python comment from johnDanger, we read the file as a binary file first then we 
            im = plt.imread(pgmf)
            imgs.append(im)

    pathFeatures = path + "/facial_features/"
    fileList = os.listdir(pathFeatures)

    for file in fileList:
        with open(pathFeatures + file, 'rb') as pgmf:
            im = plt.imread(pgmf)
            feats.append(im)

    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''
    fsize = (fsize,fsize)
    filter = np.zeros(fsize)
    radius_x = int(fsize[1]/2) if fsize[1]%2 == 1 else (fsize[1]/2-0.5)
    radius_y = int(fsize[0]/2) if fsize[0]%2 == 1 else (fsize[0]/2-0.5)
    sum = 0

    for x in np.linspace(-radius_x, radius_x, fsize[1]):
        for y in np.linspace(-radius_y, radius_y, fsize[0]):
            val = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
            filter[int(y+radius_y)][int(x+radius_x)] = val
            sum += val
    filter /= sum
    return filter

def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    downsample = x[::factor,::factor]

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = [img]
    kernel = gaussian_kernel(fsize, sigma)
    for i in range(1,nlevels):
        downsample = convolve2d(GP[i-1],kernel,mode='same')
        downsample = downsample_x2(downsample)
        GP.append(downsample)

    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''

    v1_norm = np.linalg.norm(v1,axis=1,keepdims=True)
    v2_norm = np.linalg.norm(v2,axis=1,keepdims=True)


    
    dot_product = np.dot(v1,v2.T)/v1_norm*v2_norm

    ssd = np.sum(pow(v1-v2,2))

    distance = ssd #choice = SSD
    
    return distance


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''
    min_score = None

    imgDim = img.shape
    featDim = feat.shape

    xImgDim = imgDim[0]
    yImgDim = imgDim[1]

    xFeatDim = featDim[0]
    yFeatDim = featDim[1]

    xStart = 0
    yStart = 0



    
    while(yStart + yFeatDim <= yImgDim):
        while(xStart + xFeatDim <= xImgDim):
            currentWindow = img[xStart:xStart+xFeatDim,yStart:yStart+yFeatDim] #We select the x and y range according to feat size
            currentDistance = template_distance(currentWindow,feat)/100
            if(min_score == None or currentDistance < min_score):
                min_score = currentDistance
            xStart += step #Step = 1
        xStart = 0 #We reset xStart 
        yStart += step #Step = 1
    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return ('I will use ' + self.METHODS[2] + ' because ' + self.REASONING[1])  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    (score, g_im, feat) = (None, None, None)

    for feature in feats:

        score = None
        g_im = None
        feat = None

        for image in imgs:
            pyramids = gaussian_pyramid(image, 2, 5, 1.4)

            for scaledImage in pyramids:
                currentScore = sliding_window(scaledImage, feature)
                if(score == None or currentScore < score ):
                    score = currentScore
                    feat = feature
                    g_im = scaledImage

        match.append((score, g_im, feat))

    return match