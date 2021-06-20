import numpy as np
import cv2
import geometry_vision as gv

def proj_calib_parameter(path):
    calib = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    proj_intr = calib.getNode('proj_int').mat()
    proj_dist = calib.getNode('proj_dist').mat()
    rotation = calib.getNode('rotation').mat()
    translation  = calib.getNode('translation').mat()
    t_ext = gv.transformation_r_t(rotation,np.reshape(translation,3))
    proj_res = [1920, 1080]
    return proj_intr, proj_dist, t_ext, proj_res
def zivid_parameters():
    cam_int = np.array([[1782.09204101562, 0, 977.639282226562],
                    [0, 1782.05212402344, 587.777648925781],    #read intrinsics 
                    [0,0,1]])
    cam_dist = np.array([-0.0907880067825317, 0.134410485625267, -0.0652082785964012, 0.000578985665924847, -5.82622851652559e-05]) # read distortion
    cam_res = [1944, 1200]
    return cam_int, cam_dist, cam_res 