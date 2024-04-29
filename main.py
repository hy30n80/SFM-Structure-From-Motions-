from sfm_module import SFM
import pdb


if __name__=="__main__":

    img_path = "./data_2_sfm/"
    img1_name, img2_name = "sfm01.JPG", "sfm02.JPG"

    sfm = SFM()
    img1, img2 = sfm.load_image(img_path, img1_name, img2_name)
    matches_good, img1_kp, img2_kp = sfm.SIFT(img1, img2)

    E, p1_inlier, p2_inlier = sfm.Estimation_E(matches_good, img1_kp, img2_kp)
    # CameraMatrix = sfm.EM_Decomposition(E, p1_inlier, p2_inlier)
    # Rt0, Rt1 = sfm.initialize_CM(CameraMatrix)
    # p1, p2 = sfm.rescale_point(p1_inlier, p2_inlier, len(p1_inlier))
    # point3d = sfm.make_3dpoint(p1, p2)
    # sfm.visualize_3d(point3d)


    
