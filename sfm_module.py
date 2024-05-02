import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import pdb
import matlab.engine


class SFM():
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r'C:\app\SfM', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
        self.intrinsic = np.array([
            [1698.873755, 0.000000, 971.7497705],
            [0.000000, 1698.8796645, 647.7488275],
            [0.000000, 0.000000, 1.000000]
        ])
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
        #plt.figure(figsize=(15,15))
        #plt.imshow(res)
        #plt.show()
        return matches_good, img1_kp, img2_kp



    def randomsample(self, p1, p2):
        p1p2 = np.concatenate((p1,p2), axis=1)
        random_idx = np.random.randint(p1p2.shape[0], size=5)
        p1p2_ = p1p2[random_idx,:]

        p1s = p1p2_[:,:3]
        p2s = p1p2_[:,3:]
        return p1s, p2s


    def calibrated_fivepoint(self, p1s, p2s):
        p1s_tp = p1s.T.tolist()
        p2s_tp = p2s.T.tolist()

        #pdb.set_trace()
        p1s_tp = matlab.double(p1s_tp) # a = (3,5)
        p2s_tp = matlab.double(p2s_tp) # b = (3,5)

        E = self.eng.calibrated_fivepoint(p1s_tp, p2s_tp)
        E = np.asarray(E).T # (?, 9)
        #pdb.set_trace()
        return E

    def normalization_pixel(self, p1, p2):
        p1_normalized = np.linalg.inv(self.intrinsic)@p1.T
        p2_normalized = np.linalg.inv(self.intrinsic)@p2.T

        p1_normalized = p1_normalized.T
        p2_normalized = p2_normalized.T

        return p1_normalized, p2_normalized

    def Estimation_E(self, matches_good, img1_kp, img2_kp):
        query_idx = [match.queryIdx for match in matches_good] #1370
        train_idx = [match.trainIdx for match in matches_good] #1370

        p1 = np.float32([img1_kp[ind].pt for ind in query_idx]) # (1370,2)
        p2 = np.float32([img2_kp[ind].pt for ind in train_idx]) # (1370,2)

        ones_column = np.ones((p1.shape[0], 1), dtype=p1.dtype)

        #rescale to homogeneous coordinate (x,y,1)
        p1 = np.concatenate((p1, ones_column), axis=1)
        p2 = np.concatenate((p2, ones_column), axis=1)
        
        #p1, p2 = self.normalization_pixel(p1, p2)

        iterations = 1000
        best_E =  np.random.rand(3,3)
        best_inlier_cnt = 0
        threshold = 0.1
        #threshold = 5e-7
        for i in range(iterations):
            #pdb.set_trace()
            p1s, p2s = self.randomsample(p1, p2)

            E_list = self.calibrated_fivepoint(p1s, p2s)

            for E in E_list:
                err_matrix = np.dot(np.dot(p2, E.reshape(3,3)), p1.T)
                err_list = err_matrix.diagonal()
                #pdb.set_trace()
                mask = np.where(np.abs(err_list) < threshold, 1, 0)
                curr_inlier_cnt = np.sum(mask)

                #pdb.set_trace()
                if curr_inlier_cnt > best_inlier_cnt:
                    best_E = E
                    best_inlier_cnt = curr_inlier_cnt
                    p1_inlier = p1[mask == 1]
                    p2_inlier = p2[mask == 1]

            

        #pdb.set_trace()
        return best_E, best_inlier_cnt, p1_inlier, p2_inlier
            


    def EM_Decomposition(self, E, p1_inlier, p2_inlier):
        E = E.reshape(3,3)
        U, S, VT = np.linalg.svd(E, full_matrices=True)
        W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        #Camera Matrix candidates (4)
        camera_matrix = np.array([
            np.column_stack((U @ W @ VT, U[:,2])),
            np.column_stack((U @ W @ VT, -U[:,2])),
            np.column_stack((U @ W.T @ VT, U[:,2])),
            np.column_stack((U @ W.T @ VT, -U[:,2])),
        ])

        EM_cnt = [0, 0, 0, 0]
        EM0 = np.append(np.eye(3), np.zeros((3,1)), axis=1)
        E_idx = None
        CM = np.array([[]])

        p1_normalized, p2_normalized = self.normalization_pixel(p1_inlier, p2_inlier)

        #pdb.set_trace()
        for i in range(4): 
            EM1 = camera_matrix[i]
            for k in range(len(p1_normalized)):
                #pdb.set_trace()
                A = np.array([
                    p1_normalized[k][0]*EM0[2] - EM0[0],
                    p1_normalized[k][1]*EM0[2] - EM0[1],
                    p2_normalized[k][0]*EM1[2] - EM1[0],
                    p2_normalized[k][1]*EM1[2] - EM1[1]
                ])

                U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
                X = V_A[3] / V_A[3,3]
                if X[2] > 0 and (EM1@X.T)[2] > 0:
                    EM_cnt[i] += 1
        
        E_idx = np.argmax(EM_cnt)
        EM1 = camera_matrix[E_idx]
        return EM0, EM1, p1_normalized, p2_normalized



    #Triangulation
    def LinearTriangulation(self, EM0, EM1, p1_normalized, p2_normalized):
        p3d = []
        for k in range(len(p1_normalized)):
            A = np.array([
                p1_normalized[k][0]*EM0[2] - EM0[0],
                p1_normalized[k][1]*EM0[2] - EM0[1],
                p2_normalized[k][0]*EM1[2] - EM1[0],
                p2_normalized[k][1]*EM1[2] - EM1[1]
            ])
            U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
            X = V_A[3]/V_A[3, 3]
            p3d.append(X)
        
        p3d = np.array(p3d)
        return p3d


    
    def visualize_3d(self, p3ds):
        X = np.array([])
        Y = np.array([])
        Z = np.array([])
        X = np.concatenate((X, p3ds[0]))
        Y = np.concatenate((Y, p3ds[1]))
        Z = np.concatenate((Z, p3ds[2]))

        fig = plt.figure(figsize=(15,15))
        ax = plt.axes(projection='3d')
        ax.scatter3D(X, Y, Z, c='b', marker='o')
        plt.show()


