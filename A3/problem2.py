import numpy as np


class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """

        distances = np.array([[np.sum((x-y)**2) for y in np.transpose(features1)] for x in np.transpose(features2)])
        return distances

    

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """

        
        m = p1.shape[0] 
        n = p2.shape[0] 


        # assure that p1 is the smaller set
        if m > n: 
            p1, p2 = p2, p1

        axis = 1 if n<=m else 0 
        correspondences = np.argmin(distances, axis=axis)
        pairs = np.hstack((p1, p2[correspondences]))        
        return pairs



    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """

        dimension = p1.shape[0]

        choice = np.random.choice(dimension, k, replace=False)

        sample1 = p1[choice]
        sample2 = p2[choice]


        return sample1, sample2
        



    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """

        tx, ty = np.mean(points, axis=0)
        sx, sy = 0.5 * np.max(np.abs(points), axis=0)

        T = np.array([  [1/sx,    0, -tx/sx], 
                        [   0, 1/sy, -ty/sy], 
                        [   0,    0,      1]
                    ])
        
        ps = np.hstack((points, np.ones((points.shape[0], 1))))
        ps = ps @ np.transpose(T)

        return ps, T


    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

        l = p1.shape[0]

        # construct A
        A = np.zeros((l*2, 9))

        for i in range(0,l):
            x, y, _ = p1[i]
            x_, y_, _ = p2[i]

            A[i*2  ,:] = np.array([0, 0, 0, x, y, 1, -x*y_, -y*y_, -y_])
            A[i*2+1,:] = np.array([-x, -y, -1, 0, 0, 0, x*x_, y*x_, x_])
        
        _, _, VT = np.linalg.svd(A)

        V = np.transpose(VT)

        HC = V[:, -1:]

        HC = np.reshape(HC, (3,3))
        H = np.linalg.inv(T2) @ HC @ T1

        HC = 1/HC[2,2] * HC   # normalize
        H = 1/H[2,2] * H  #normalize

        return H, HC


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

        p_hom = np.hstack((p, np.ones((p.shape[0], 1), dtype='double')))

        p_hom = p_hom @ np.transpose(H)

        # to avoid division by zero we add a small error to z-components with value 0
        p_hom[:,-1:][p_hom[:,-1:] == 0] = 1e-20

        points = p_hom[:, :2] / p_hom[:, -1:]

        return points


    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        sum1 = np.sum((self.transform_pts(p1, H) - p2)**2, axis=1)
        sum2 = np.sum((p1 - self.transform_pts(p2, np.linalg.pinv(H)))**2, axis=1)
        dist = sum1 + sum2
        return dist


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        inliers = pairs[dist < threshold]
        N = inliers.shape[0]

        return N, inliers


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        
        return int(np.ceil(np.log(1-z)/np.log(1-p**k)))




    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
        max_inliers_N = 0
        max_inliers = 0
        max_H = np.empty((3,3))

        p1 = pairs[:,:2]
        p2 = pairs[:,-2:]


        for it in range(0, int(n_iters)):
            # draw a sample of k corresponding point pairs
            s1, s2 = self.pick_samples(p1, p2, k)

            # condition sample points
            s1_cond, T1 = self.condition_points(s1)
            s2_cond, T2 = self.condition_points(s2)

            # estimate homography
            H, _ = self.compute_homography(s1_cond, s2_cond, T1, T2)
            
            # evaluate homography
            dist = self.compute_homography_distance(H, p1, p2)

            # compute inliers
            N, inliers = self.find_inliers(pairs, dist, threshold)

            if N > max_inliers_N:
                max_inliers_N = N
                max_inliers = inliers
                max_H = H

        return max_H, max_inliers_N, max_inliers


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        p1, T1 = self.condition_points(inliers[:, :2])
        p2, T2 = self.condition_points(inliers[:,-2:])

        H, HC = self.compute_homography(p1, p2, T1, T2)

        return H