import numpy as np
from numpy.core.fromnumeric import transpose
import scipy.linalg


def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate 
        world: A N*3 array of 3D points form world coordinate
    '''
    points = np.load(path)
    image = points['image.npy']
    world = points['world.npy']

    return image,world

def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0]
    
    A = np.zeros((2*N, 12))

    #According to slide 51 of lecture 2, we will build A matrix 2 line by 2 line, at the moment A is a (2*N,12) matrix filled up with 0 everywhere 
    for i in range(0, 2*N, 2):
        Xi = np.transpose(X[i//2]) #We are using transpose(Xi) everywhere so let's keep it in a variable
        xi = x[i//2][0] # // Because we need integer there
        yi = x[i//2][1]

        #First line
        #Zeros values are already presents so we don't reattribute them 
        A[i][4] = -Xi[0]
        A[i][5] = -Xi[1]
        A[i][6] = -Xi[2]
        A[i][7] = -Xi[3]
        A[i][8] = yi*Xi[0]
        A[i][9] = yi*Xi[1]
        A[i][10] = yi*Xi[2]
        A[i][11] = yi*Xi[3]

        #Second line
        A[i+1][0] = Xi[0]
        A[i+1][1] = Xi[1]
        A[i+1][2] = Xi[2]
        A[i+1][3] = Xi[3]
        A[i+1][8] = -xi*Xi[0]
        A[i+1][9] = -xi*Xi[1]
        A[i+1][10] = -xi*Xi[2]
        A[i+1][11] = -xi*Xi[3]

    return A

def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """

    At = np.transpose(A)
    AtA = np.dot(At,A)

    #We will perform a SVD on AtA, and take last vector of v according to the lecture (smallest eingenvalue for AtA)

    u, s, v = np.linalg.svd(AtA, full_matrices=False) #true_matrices = False because U -> m*n and V -> n*n

    vT = np.transpose(v)
    lastRightSingularValue = vT[:,-1] #To find the eigenvector of ATA with the smallest eigenvalue, we compute the last right-singular vector of A. (slide 56)
    P = np.reshape(lastRightSingularValue, (3,4))

    #Proof that P is good
    #print("LASTSINGULAR VALUE : ", lastRightSingularValue)
    #print(s)
    #print(np.dot(AtA,np.reshape(P,(12,1)))[0]/np.reshape(P,(12,1))[0])
    #print("Ap = ", np.dot(A,np.reshape(P,(12,1)))) #Because Ap where P is a column vector (check slide 50 and 51 pT and p)

    return P


def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix 
    """

    #According to the lecture (slide 58), we assume that P is a 3x4 matrix, M a 3x3 matrix and m a 3x1 vector

    M = P[:,0:3] #We select the first 3 columns of P

    
    K, R = scipy.linalg.rq(M) # Decomposition RQ -> Here it's Decomposition KR With K upper triangular and R orthonormal
    
    #Hint 2 -> Multiply K by 1/(value right bottom corner) then extract scalar et apply it on R
    scalar = K[2][2]
    K = K/scalar
    R = R*scalar    
    
    #Hint 3 -> Ensure positive elements in K, we have to check the diagonal of K and ensure all elements are positive, last element is 1 so we don't need to check it
    #If we find a negative element then we multiply it by -1 and we change signs accordingly in R to get the same result with KR 
    
    for i in range(2):
        if K[i][i] < 0:
            K[i][i] = K[i][i]*-1 #We inverse sign to get a positive focal distance
            R[0][i] = R[0][i]*-1 #We change sign column signs of R to keep the same KR (always inverse sign in R because K[i][i] element was negative)
            R[1][i] = R[1][i]*-1
            R[2][i] = R[2][i]*-1

    return K,R


def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """

    u, s, v = np.linalg.svd(P, full_matrices=True) #true_matrices = True because U -> m*m and V -> n*n

    HomogeneousC = (v[-1,:])

    c = np.zeros((3,1))
    c[0] = HomogeneousC[0]/HomogeneousC[3] #Let's go to the non-homogeneous form !
    c[1] = HomogeneousC[1]/HomogeneousC[3]
    c[2] = HomogeneousC[2]/HomogeneousC[3]

    return c
