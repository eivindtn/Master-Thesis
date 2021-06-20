import numpy as np
import cv2
import os
import zivid
from scipy.interpolate import griddata
import glob
import random
np.set_printoptions(suppress=True)



class ProjectorCalibrate:

    def __init__(self, cols, rows, images, zdf, camera_intrinsics, distortion_coeffsients, projector_res_width, projector_res_height,method = 'corner', images_to_iterate = None, flags= None, number_shuffled_iteration = 3, number_under_len = 3, square_size = 90, interpolate_method = 'nearest', interpolate_number=20):
        self.cols = cols
        self.rows = rows
        self.images = images
        self.images_to_iterate = len(images) if images_to_iterate == None else images_to_iterate
        self.zdf = zdf
        self.method = method
        self.camera_intrinsics = camera_intrinsics
        self.distortion_coeffsients = distortion_coeffsients
        self.flags = flags
        self.checkerboard = (cols, rows)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.projector_res_width = projector_res_width
        self.projector_res_height = projector_res_height
        self.square_size = square_size
        self.interpolate_method = interpolate_method
        self.interpolate_number = interpolate_number
        self.number_shuffled_iteration = number_shuffled_iteration
        self.number_under_len = number_under_len

    def Find_Chess_Board_Corners(self, fname):
        img = cv2.imread(self.images[fname])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), self.criteria)
            
            return True, corners2, img 
        if ret == False:
            print("No found corners in frame: ", self.images[fname])
            return False

    def Load_Zdf_Frames(self, fname):
        app = zivid.Application()
        frame = zivid.Frame(self.zdf[fname])
        xyz = frame.point_cloud().copy_data("xyz")
        return xyz

    def line_line_intersect(self,p1, p2, p3, p4):
        p1 = p1.ravel()
        p2 = p2.ravel() #flatten the arrays
        p3 = p3.ravel()
        p4 = p4.ravel()
        # finds the point of intersection of two lines l1=(p1, p2), l2=(p3,p4)
        denominator = (p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0])
        point = [((p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0])-(p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0]))/denominator,((p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0]))/denominator]
        return point

    def Find_Center_Of_4_Coordinates(self, array):
        center = np.array([])
        center_byline = np.array([])
        for k in range(0,self.rows*self.cols-self.cols,self.cols):
            for i in range(0,self.cols-1):
                center = np.append(center, ([array[i+k], array[i+1+k], array[i+self.cols+k],array[i+self.cols+1+k]]))
                center_byline  = np.append(center_byline, ([self.line_line_intersect(array[i+k], array[i+self.cols+1+k], array[i+1+k], array[i+self.cols+k])]))
        center = np.reshape(center, ((self.rows-1)*(self.cols-1),4,2))
        center_byline = np.reshape(center_byline, ((self.rows-1)*(self.cols-1),2))
        center = np.average(center.T,axis = 1)
        return center.T

    def Save_Projector_Image_Coordinates(self):
        checkerboardcordinates = []
        xspacing = (self.projector_res_width - (self.cols+1) * self.square_size) / 2.0
        yspacing = (self.projector_res_height - (self.rows+1) * self.square_size) / 2.0
        for y in range(int(yspacing),self.projector_res_height-int(yspacing), self.square_size):
            if (y + self.square_size > self.projector_res_height -int(yspacing)):
                break
            for x in range(int(xspacing), self.projector_res_width-int(xspacing), self.square_size):
                if (x + self.square_size > self.projector_res_width -int(xspacing)):
                    break
                if(x > int(xspacing) and y > int(yspacing)):   
                    checkerboardcordinates = np.append(checkerboardcordinates, np.array([x,y]))
        checkerboardcordinates = np.reshape(checkerboardcordinates, (-1,2))
        #print(checkerboardcordinates)
        return checkerboardcordinates
    
    def Interpolate_xyz(self, array, u, v):
        xyz = array[v-self.interpolate_number:v+self.interpolate_number+1, u-self.interpolate_number:u+self.interpolate_number+1]
        x_indx,y_indx,z_indx = np.meshgrid(np.arange(0, np.shape(xyz)[0]),np.arange(0,np.shape(xyz)[1]), np.arange(0,np.shape(xyz)[2]))
            
        array_masked = np.ma.masked_invalid(xyz)
        valid_xs = x_indx[~array_masked.mask]
        valid_ys = y_indx[~array_masked.mask]
        valid_zs = z_indx[~array_masked.mask]
        validarr = array_masked[~array_masked.mask]
            
        xyz_interp = griddata((valid_xs, valid_ys, valid_zs), validarr.ravel(),
                                        (x_indx, y_indx, z_indx), method=self.interpolate_method)
        return xyz_interp[self.interpolate_number][self.interpolate_number]

    def Re_Projection_Error(self,imagepoints,objectpoints,rvecs,tvecs, mtx,dist):
        imagepoints = np.reshape(imagepoints, (len(imagepoints),len(imagepoints[0]),1,len(imagepoints[0][0])))
        mean_error = 0
        tot_mean_error= 0
        rms_tot_error = 0
        total_points = 0
        rms_mean_error = 0 
        Re_projection_frames = []
        for i in range(len(objectpoints)):
            imgpoints2, _ = cv2.projectPoints(objectpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imagepoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
            Re_projection_frames.append([i+1,error])
            rms_tot_error+=np.sum(np.abs(imagepoints[i]-imgpoints2)**2)
            total_points+=len(objectpoints[i])
        rms_mean_error=np.sqrt(rms_tot_error/total_points) 
        tot_mean_error = mean_error/len(objectpoints) 
        return tot_mean_error, Re_projection_frames


    def ProjectorCalibrate(self):
        P_p_2d = []  
        P_c_2d = []  
        P_3d   = []         
        P_3d_all = []
        P_3d_obj = []
        nan_pixels = 0
        
        for fname in range (0, self.images_to_iterate):
            P_3d = []
            corners2 = self.Find_Chess_Board_Corners(fname)[1]
            xyz = self.Load_Zdf_Frames(fname)
            if self.Find_Chess_Board_Corners(fname)[0] == True:
                print("Found corners in frame: ", self.images[fname])
                if self.method == "center":
                    corners2 = self.Find_Center_Of_4_Coordinates(corners2)
                    corners2 = np.reshape(corners2,((self.cols-1)*(self.rows-1),1,2))
                    P_p_2d.append(self.Find_Center_Of_4_Coordinates(self.Save_Projector_Image_Coordinates()))
                elif self.method == "corner":
                    P_p_2d.append(self.Save_Projector_Image_Coordinates())
                else:
                    print("Wrong method: Try center or corner.")
                P_c_2d.append(corners2)
                for i in range(len(corners2)):
                    if np.isnan(xyz[int(corners2[i][0][1])][int(corners2[i][0][0])]).any() == True:
                        nan_pixels += 1
                        P_3d.append(self.Interpolate_xyz(xyz,int(corners2[i][0][0]),int(corners2[i][0][1])))
                    else:
                        P_3d.append(xyz[int(corners2[i][0][1])][int(corners2[i][0][0])])

            P_3d_all.append(P_3d)
            P_3d_t = np.transpose(P_3d)
            centroid = np.average(P_3d_t,axis = 1)
            P_3d_c = (P_3d_t.T-np.average(P_3d_t,axis = 1)).T
            u, s, vh = np.linalg.svd(P_3d_c)
            P_3d_rotate_t = np.dot(np.transpose(u), P_3d_c)
            P_3d_obj.append(P_3d_rotate_t.T)

        P_3d_obj_plan = P_3d_obj
        
        P_3d_obj_plan = np.asarray(P_3d_obj_plan)
        P_3d_all = np.asarray(P_3d_all)
        P_3d_obj = np.asarray(P_3d_obj)
        P_p_2d = np.asarray(P_p_2d)
        P_c_2d = np.asarray(P_c_2d)
        
        P_3d_obj[:,:,2] = 0

        P_3d_all = P_3d_all.astype('float32')
        P_3d_obj = P_3d_obj.astype('float32')
        P_p_2d = P_p_2d.astype('float32')
        P_c_2d = P_c_2d.astype('float32')
        
        result_flag = []
        result_noflag = []
        index = []

        for i in range (self.images_to_iterate-self.number_under_len, self.images_to_iterate):
            for j in range(1,self.number_shuffled_iteration):
                P_3d_obj_shuffle, P_c_2d_shuffle, P_p_2d_shuffle  = zip(*random.sample(list(zip(P_3d_obj, P_c_2d, P_p_2d)), i))
        
                mask = (np.isin(P_3d_obj, P_3d_obj_shuffle))
                
                for k in range(0, len(mask)):
                    if mask[k].all():
                        index.append([i,j,self.images[k]])

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(P_3d_obj_shuffle, P_p_2d_shuffle,(self.projector_res_width,self.projector_res_height), None, None, None, None, self.flags, self.criteria)
                retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(P_3d_obj_shuffle, P_c_2d_shuffle, P_p_2d_shuffle, self.camera_intrinsics, self.distortion_coeffsients, mtx, dist, (self.projector_res_width,self.projector_res_height))
                
                print("With OpenCvs flag: ","Number of extracted frames: ", (i+1),"iteration No: ", j, ret,retval)

                result_flag.append([ret, retval, mtx,dist, self.Re_Projection_Error(P_p_2d_shuffle,P_3d_obj_shuffle,rvecs,tvecs, mtx,dist)[0],self.Re_Projection_Error(P_p_2d_shuffle,P_3d_obj_shuffle,rvecs,tvecs, mtx,dist)[1], R, T, (i+1), j])

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(P_3d_obj_shuffle, P_p_2d_shuffle,(self.projector_res_width,self.projector_res_height), None, None, None, None, None, self.criteria)
                retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(P_3d_obj_shuffle, P_c_2d_shuffle, P_p_2d_shuffle, self.camera_intrinsics, self.distortion_coeffsients, mtx, dist, (self.projector_res_width,self.projector_res_height))
                
                print("With no OpenCvs flag: ","Number of extracted frames: ", (i+1),"iteration No: ", j, ret,retval)

                result_noflag.append([ret,retval, mtx,dist, self.Re_Projection_Error(P_p_2d_shuffle,P_3d_obj_shuffle,rvecs,tvecs, mtx,dist)[0],self.Re_Projection_Error(P_p_2d_shuffle,P_3d_obj_shuffle,rvecs,tvecs, mtx,dist)[1], R, T, (i+1), j])

        result_flag  = np.reshape(result_flag,(self.number_under_len,self.number_shuffled_iteration-1,10))  
        result_noflag  = np.reshape(result_noflag,(self.number_under_len,self.number_shuffled_iteration-1,10))
        best_result_flag = []
        best_result_noflag = []
        for i in range (0, self.number_under_len):
            ret_projcalib, ret_stereocalib, proj_intrinsics, proj_dist, mean_reprojection,reprojection_in_frame, Rotation_matrix, Translation_vector, no_images, iteration = zip(*sorted(zip(result_flag[:,:,0][i], result_flag[:,:,1][i], result_flag[:,:,2][i], result_flag[:,:,3][i], result_flag[:,:,4][i], result_flag[:,:,5][i],result_flag[:,:,6][i],result_flag[:,:,7][i], result_flag[:,:,8][i], result_flag[:,:,9][i])))
            best_result_flag.append([ret_projcalib[0], ret_stereocalib[0], proj_intrinsics[0], proj_dist[0], mean_reprojection[0], reprojection_in_frame[0], Rotation_matrix[0], Translation_vector[0], no_images[0], iteration[0]])

            ret_projcalib, ret_stereocalib, proj_intrinsics, proj_dist, mean_reprojection,reprojection_in_frame, Rotation_matrix, Translation_vector, no_images, iteration = zip(*sorted(zip(result_noflag[:,:,0][i], result_noflag[:,:,1][i], result_noflag[:,:,2][i], result_noflag[:,:,3][i], result_noflag[:,:,4][i], result_noflag[:,:,5][i],result_noflag[:,:,6][i],result_noflag[:,:,7][i], result_noflag[:,:,8][i], result_noflag[:,:,9][i])))
            best_result_noflag.append([ret_projcalib[0], ret_stereocalib[0], proj_intrinsics[0], proj_dist[0], mean_reprojection[0], reprojection_in_frame[0], Rotation_matrix[0], Translation_vector[0], no_images[0], iteration[0]])

        best_result_flag = np.asarray(best_result_flag)
        best_result_noflag = np.asarray(best_result_noflag)
        sorted_best_result_flag = []
        sorted_best_result_noflag = []
        
        ret_projcalib, ret_stereocalib, proj_intrinsics, proj_dist, mean_reprojection,reprojection_in_frame, Rotation_matrix, Translation_vector, no_images, iteration = zip(*sorted(zip(best_result_flag[:,0], best_result_flag[:,1], best_result_flag[:,2], best_result_flag[:,3], best_result_flag[:,4], best_result_flag[:,5], best_result_flag[:,6], best_result_flag[:,7], best_result_flag[:,8], best_result_flag[:,9] )))
        for i in range (0, self.number_under_len):
            sorted_best_result_flag.append([ret_projcalib[i], ret_stereocalib[i], proj_intrinsics[i], proj_dist[i], mean_reprojection[i], reprojection_in_frame[i], Rotation_matrix[i], Translation_vector[i], no_images[i], iteration[i]])

        ret_projcalib, ret_stereocalib, proj_intrinsics, proj_dist, mean_reprojection,reprojection_in_frame, Rotation_matrix, Translation_vector, no_images, iteration = zip(*sorted(zip(best_result_noflag[:,0], best_result_noflag[:,1], best_result_noflag[:,2], best_result_noflag[:,3], best_result_noflag[:,4], best_result_noflag[:,5], best_result_noflag[:,6], best_result_noflag[:,7], best_result_noflag[:,8], best_result_noflag[:,9] )))
        for i in range (0, self.number_under_len):    
            sorted_best_result_noflag.append([ret_projcalib[i], ret_stereocalib[i], proj_intrinsics[i], proj_dist[i], mean_reprojection[i], reprojection_in_frame[i], Rotation_matrix[i], Translation_vector[i], no_images[i], iteration[i]])


        '''ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(P_3d_obj, P_p_2d,(self.projector_res_width,self.projector_res_height), None, self.criteria)
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(P_3d_obj, P_c_2d, P_p_2d, self.camera_intrinsics, self.distortion_coeffsients, mtx, dist, (self.projector_res_width,self.projector_res_height))
        print("No flags:" , k, ret, retval, dist)'''   
        return sorted_best_result_flag, sorted_best_result_noflag, P_3d_all, P_3d_obj_plan, nan_pixels 
def main():
    #Test parameters, see explaination of parameters in "Calibrate.py"
    cols = 9
    rows = 6
    images = glob.glob('captured_images/*.png')
    zdf = glob.glob('zdf/*.zdf')
    method = 'center'
    camera_intrinsics = np.array([[2767.193359, 0 , 942.942505], [0, 2766.449119, 633.898],[0, 0, 1]])
    distortion_coeffsients = np.array([[-0.26513, 0.282772, 0.000767328, 0.000367037,0]])
    flags = cv2.CALIB_ZERO_TANGENT_DIST
    projector_res_width = 1024
    projector_res_height = 768
    square_size = 90
    interpolate_method = 'nearest'
    interpolate_number = 50
    images_to_iterate = 21
    number_shuffled_iteration = 15
    number_under_len = 6

    Calib = ProjectorCalibrate(cols, rows, images, zdf, camera_intrinsics, distortion_coeffsients, projector_res_width, projector_res_height,method, images_to_iterate, flags, number_shuffled_iteration, number_under_len, square_size, interpolate_method, interpolate_number)
    
    Calib.ProjectorCalibrate()


if __name__ == "__main__":

    main()


