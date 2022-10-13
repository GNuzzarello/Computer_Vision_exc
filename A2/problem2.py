import numpy as np
import os
from PIL import Image

def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    image_list = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            img = Image.open(os.path.join(root, name))
            elem = np.reshape(np.array(img.getdata(), dtype=np.float64), (img.height, img.width))
            image_list.append(elem)

    images = np.array(image_list)
    N = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    M = H * W


    x = np.reshape(images, (N,M))

    return x, (H, W)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (self.METHODS[1], self.REASONING[2], self.REASONING[4])

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """
    
    N = X.shape[0]

    mean = np.mean(X, axis=0)

    X_hat = np.transpose(X - mean)


    U, S, V = np.linalg.svd(X_hat)

    lmb = []
    for i in range(0, S.shape[0]):
        # variance = 1/N * s^2
        var_i = 1/N*S[i]**2
        lmb.append(var_i)

    return U, lmb

#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """

    sum_lambda = p * np.sum(s)
    sum = 0.0
    D = 0
    while sum < sum_lambda:
        sum += s[D]
        D += 1 

    D += 1
    return u[:,0:D]
    
#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """
    a = np.transpose(u) @ (face_image) #assume face_images normalized => face_images = face_image - mean
    image_out = np.sum(a * u, axis=1)

    return image_out

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (self.OBSERVATION[1])


#
# Task 7
#


def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """
    N = Y.shape[0]
    M = Y.shape[1]

    a_x = np.transpose(u) @ (x)

    sim = np.zeros(N)
    for i in range(0, N):
        a_y = np.transpose(u) @ Y[i]
        sim[i] = np.linalg.norm(a_x - a_y)

    sim_asc_ind = np.argsort(sim)

    Y_ = np.zeros((top_n, M))
    for i in range(0, top_n):
        Y_[i] = Y[sim_asc_ind[-(i + 1)]]

    return Y_

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    
    M = x1.shape[0]

    x1_a = np.transpose(u) @ (x1)
    x2_a = np.transpose(u) @ (x2)
    delta = x2_a - x1_a

    Y = np.zeros((N, M))
    ipol = np.linspace(0, 1, N + 2)

    for i in range(0, N):
        a = x1_a + ipol[i + 1] * delta
        Y[i] = np.sum(a * u, axis=1)


    return Y
