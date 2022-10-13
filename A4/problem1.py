import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scp

def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    #
    # You code here
    #

    Uf, Dftilt, VfT = np.linalg.svd(A)

    Dftilt = np.diag(Dftilt) #Need to diagonalize Dftilt bc atm it's a vector
    Dftilt[2][2] = 0
    Aconditionned = np.dot(Uf, np.dot(Dftilt,VfT))

    return Aconditionned




def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    #
    # You code here
    #
    n = p1.shape[0]

    A = []

    for index in range(n):
        x = p1[index][0]
        y = p1[index][1]
        z = p1[index][2]

        xp = p2[index][0]
        yp = p2[index][1]
        zp = p2[index][2]

        x = x/z #Go to non-homogeneous form
        y = y/z

        xp = xp/zp
        yp = yp/zp

        A.append([x*xp, y*xp, xp, x*yp, y*yp, yp, x, y, 1])
        
    A = np.asarray(A)
        
    u,s,vt = np.linalg.svd(A)
    EstimatedF = vt[-1,:]
    EstimatedF = np.reshape(EstimatedF,(3,3))
    
    Fconditionned = enforce_rank2(EstimatedF)

    return Fconditionned

def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    #
    # You code here
    #
    p1, T1 = condition_points(p1)
    p2, T2 = condition_points(p2)

    Fconditionned = compute_fundamental(p1, p2)
    Funconditionned = np.dot(np.transpose(T2),np.dot(Fconditionned, T1))

    return Funconditionned


def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """

    #
    # You code here
    #
    n = p1.shape[0]
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []

    xmax = img.shape[1] - 1

    #Line equation ax + by + c = 0, with l = Fx from lecture we will get line equation, then find point on line for x = 0 and x = xmax (left and right border of the image)

    p1Homogeneous = np.c_[p1,np.ones(n)] #Add one column full of 1 

    for i in range(n):
        lineEq = np.dot(F,p1Homogeneous[i])
        lineEq = np.transpose(lineEq)
        lineEq = lineEq/lineEq[2]   #Go back to non-homogeneous WARNING : I don't know if it is really required, 
                                    #but we are looking for a 2D equation line, so 3D could make the result differ a bit (according to numerical situation, I checked both and saw no differences)
        a = lineEq[0]
        b = lineEq[1]
        c = lineEq[2] #Should be equal to one


        X1.append(0)
        X2.append(xmax)

        Y1.append(-c/b) #Case where x = 0
        Y2.append(((-a*xmax)-c)/b) #Case where x = xmax

    return X1, X2, Y1, Y2




def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    #
    # You code here
    #

    residuals = []
    n = p1.shape[0] #We assume p1.shape == p2.shape

    for i in range(n):
        residuals.append(np.absolute( np.dot( p1[i], np.dot( F, np.transpose(p2[i]) ) ) ) ) #We transpose p2 instead of p1 because row/column vectors are reversed compared to the exercise sheet in numpy

    residuals = np.asarray(residuals)
    max_residual = np.amax(residuals)
    avg_residual = np.mean(residuals)

    return max_residual, avg_residual

def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    #
    # You code here
    #

    #According to the lecture, Fe1 = 0 and transpose(F)e2 = 0
    e1 = scp.null_space(F)
    e2= scp.null_space(np.transpose(F))
    
    e1 = e1/e1[2] #Back to non-homogeneous
    e2 = e2/e2[2]

    e1 = e1[0:2] #We slice to keep only x and y coordinates 
    e2 = e2[0:2]


    #Check if it works, I don't know yet if we are allowed to use scipy.linalg, but it works yet
    #print("F * nullspace(F) :", np.dot(F, e1 ))
    #print("nullspace(FT) * FT :", np.dot(np.transpose(F), e2 ) )

    return e1, e2
