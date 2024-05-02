from sfm_module import SFM
import pdb
import cv2


if __name__=="__main__":

    img_path = "./data_2_sfm/"
    img1_name, img2_name = "sfm01.JPG", "sfm02.JPG"

    sfm = SFM()
    img1, img2 = sfm.load_image(img_path, img1_name, img2_name)
    matches_good, img1_kp, img2_kp = sfm.SIFT(img1, img2)
    E, inlier_cnt, p1_inlier, p2_inlier = sfm.Estimation_E(matches_good, img1_kp, img2_kp)
    #pdb.set_trace()
    EM0, EM1, p1_normalized, p2_normalized = sfm.EM_Decomposition(E, p1_inlier, p2_inlier)
    #pdb.set_trace()
    point3d  = sfm.LinearTriangulation(EM0, EM1, p1_normalized, p2_normalized)
    sfm.visualize_3d(point3d)


    
