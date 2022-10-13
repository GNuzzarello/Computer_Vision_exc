import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """

  #As we saw in lecture 4, the gaussian filter give more weight to values near from the "center" of the filter, so we need to define it
  #We don't know how to deal with cases like 2*3 kernel (cannot find a real "center" of the matrix to set G(0,0), so it will not be centred, we'll use //2 for every case)
  #We assume fsize has the form (x,y) (two dimensions given, even if x or y equals 1 which is column/row matrix)

  #We are gonna compute G(x,y) everytime, so let's define it as a function :
  def G(x,y):
    return (1/2*np.pi*pow(sigma,2))*np.exp( (-(pow(x,2)+pow(y,2)) ) / (2*pow(sigma,2)) )
  
  g = np.zeros(fsize)

  xCenter = fsize[0]//2
  yCenter = fsize[1]//2

  for x in range(fsize[0]):
    for y in range(fsize[1]):
      g[x][y] = G(x-xCenter,y-yCenter) #We apply G(x,y) compute according to the "center" of our filter

  return g


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """

  sigma = 0.9 
  y_gauss = gauss2d(sigma, (3, 1)) 
  x_diff = np.array([[0.5, 0, -0.5]]) #according to slide 41, 1/2 * [1,0,-1] 
  fx = np.zeros(shape=(3,3)) 
 
  x_gauss = gauss2d(sigma, (1, 3)) 
  y_diff = np.transpose(np.array([[0.5, 0, -0.5]])) 
  fy = np.zeros(shape=(3,3)) 
 
  fx = y_gauss @ x_diff 
  fy = y_diff @ x_gauss 

  return fx,fy


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  Ix = ndimage.convolve(I, fx) 
  Iy = ndimage.convolve(I, fy) 

  return Ix, Iy


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """
  edges = np.sqrt(Ix**2 + Iy**2)
  dimensions = edges.shape
  for x in range(dimensions[0]):
    for y in range(dimensions[1]):
      if(edges[x][y] < thr):
        edges[x][y] = 0


  return edges


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """

  edges2 = np.copy(edges)
  dimensions = edges.shape

  for x in range(1,dimensions[0]-1):
    for y in range(1,dimensions[1]-1):
      m = edges[x][y]
      if m > 0.0:
        theta = np.rad2deg(np.arctan2(Iy[x][y],Ix[x][y]))
        if theta > 270:
          theta = 270 - theta
        elif theta > 90:
          theta = theta - 180

  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
        if (theta>= -90 and theta <= -67.5) or (theta> 67.5 and theta <= 90):
          neighbour1 = edges[x,y-1]
          neighbour2 = edges[x,y+1]
          if neighbour1 > m or neighbour2 > m:
            edges2[x][y] = 0

  # handle left-to-right edges: theta in (-22.5, 22.5]

        if (theta> -22.5 or theta <= 22.5):
              neighbour1 = edges[x-1,y]
              neighbour2 = edges[x+1,y]
              if neighbour1 > m or neighbour2 > m:
                edges2[x][y] = 0

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

        if (theta>= 22.5 and theta <= 67.5):
              neighbour1 = edges[x-1,y-1]
              neighbour2 = edges[x+1,y+1]
              if neighbour1 > m or neighbour2 > m:
                edges2[x][y] = 0

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

        if (theta>= -67.5 and theta <= -22.5):
              neighbour1 = edges[x+1,y-1]
              neighbour2 = edges[x-1,y+1]
              if neighbour1 > m or neighbour2 > m:
                edges2[x][y] = 0

  return edges2

      
