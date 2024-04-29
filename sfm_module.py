import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import pdb

class SFM():
    def __init__(self):
        pass

    #Load image
    def load_image(self, img_path, img1_name, img2_name):
        img1 = cv2.imread(img_path + img1_name)
        img2 = cv2.imread(img_path + img2_name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        return img1, img2
    
    #Fine feature correspondence with SIFT
    def SIFT(self, img1, img2):
        # 1.Extracting keypoints from each image
        sift = cv2.xfeatures2d.SIFT_create()
        img1_kp, img1_des = sift.detectAndCompute(img1, None) #2706
        img2_kp, img2_des = sift.detectAndCompute(img2, None) #2531

        # img1_draw = cv2.drawKeypoints(img1, img1_kp, None)
        # img2_draw = cv2.drawKeypoints(img2, img2_kp, None)
        # plt.figure(figsize=(20,20))
        # plt.imshow(img1_draw)
        # plt.show()

        # 2.Match features between 2 images (Brute Force)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(img1_des, img2_des, k=2)

        matches_good = [m1 for m1, m2 in matches if m1.distance < 0.95*m2.distance] #2706 -> 1370
        sorted_matches = sorted(matches_good, key=lambda x: x.distance)
        res = cv2.drawMatches(img1, img1_kp, img2, img2_kp, sorted_matches, img2, flags=2)

        #draw matches
        plt.figure(figsize=(15,15))
        plt.imshow(res)
        plt.show()
        return matches_good, img1_kp, img2_kp



    def randomsample(self, p1, p2):
        p1p2 = np.concatenate((p1,p2), axis=1)
        random_idx = np.random.randint(p1p2.shape[0], size=5)
        p1p2_ = p1p2[random_idx,:]

        p1s = p1p2_[:,:2]
        p2s = p1p2_[:,2:]
        return p1s, p2s


    def Estimation_E(self, matches_good, img1_kp, img2_kp):
        query_idx = [match.queryIdx for match in matches_good] #1370
        train_idx = [match.trainIdx for match in matches_good] #1370

        p1 = np.float32([img1_kp[ind].pt for ind in query_idx]) # (1370,2)
        p2 = np.float32([img2_kp[ind].pt for ind in train_idx]) # (1370,2)
        

        focal = 1698.8796645
        fx, fy = 971.7497705, 647.7488275

        """ RANSAC 직접 구현
        for i in range(iterations):
            p1s, p2s = self.randomsample(p1, p2)
            pdb.set_trace()
        """

        E, mask = cv2.findEssentialMat(p1, p2, method=cv2.RANSAC, focal=focal, pp=(fx, fy), maxIters = 500, threshold=1)
        p1 = p1[mask.ravel()==1] #left image inlier
        p2 = p2[mask.ravel()==1] #right image inlier

        return E, p1, p2


    def EM_Decomposition(self, E, p1, p2):
            
        #return CM
        pass

    #Rescale to Homogeneous Coordinate
    def rescale_point(self, pts1, pts2, length):
        #return a,b
        pass

    #Intrinsic parameter K
    def initialize_CM(self, CM):
        #return Rt0, Rt1
        pass

    #Triangulation
    def LinearTriangulation(self, Rt0, Rt1, p1, p2):
        #return VT[3,0:3] / VT[3,3]
        pass


    def make_3dpoint(self, p1, p2):
        #return p3ds
        pass

    
    def visualize_3d(self, p3ds):
        X = np.array([])
        Y = np.array([])
        Z = np.array([])
        X = np.concatenate((X, p3ds[0]))
        Y = np.concatenate((Y, p3ds[1]))
        Z = np.concatenate((Z, p3ds[2]))

        fig = plt.figure(figsize=(15,15))
        ax = plt.axed(projection='3d')
        ax.scatter3D(X, Y, Z, c='b', marker='o')
        plt.show()


