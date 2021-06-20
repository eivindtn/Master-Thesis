from ProjectorCalibrate import ProjectorCalibrate
import numpy as np
import glob 
import cv2
import copy
import open3d
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def writeCSV(filename, matrix):
    np.savetxt(filename, matrix, delimiter=",", fmt='% s')

def visualizationstereo(points, rotation, translation,windowname):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    c_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
    p_frame = copy.deepcopy(c_frame)
    p_frame.rotate(rotation, center = (0,0,0))
    p_frame = copy.deepcopy(p_frame).translate((translation[0][0], translation[1][0], translation[2][0]))
    open3d.visualization.draw_geometries([point_cloud ,c_frame, p_frame],window_name=windowname,width=1000, height=1080)

def plot_reprojection(Re_projection_frames, mean_error,method):
    Re_projection_frames = np.asarray(Re_projection_frames)
    fig, ax = plt.subplots(figsize=(8,8))
    #fig.patch.set_visible(False)
    ax.scatter(Re_projection_frames[:,0] , Re_projection_frames[:,1], s=5.0, color = "b", label = "Frames")
    ax.grid(True)
    
    ax.plot(Re_projection_frames[:,0], np.full((len(Re_projection_frames,)),(mean_error)), color = "r",label = "Mean= {} px".format(np.round_(mean_error,3)), linestyle = "--") #rett linje 
    ax.set(title='Re-Projection Error {}'.format(method) , ylabel= "Re-Projection Error (px)",  xlabel='Frames')
    legend = ax.legend(loc=(0.7,-0.14), shadow=True, fontsize='large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#afeeee')

    fig.savefig('lab/calibration/results/Re-Projection-{}.png'.format(method))


def plot_frame_rms_error(labels, rms_calib, rms_stereo, title, numberimages):
    labels = np.asarray(labels).astype(int)
    rms_calib = np.asarray(rms_calib).astype('float32')
    rms_stereo = np.asarray(rms_stereo).astype('float32')

    x = np.arange(len(labels))  # the label locations
    width = 0.55  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,10))
    rects1 = ax.bar(x - width/2, rms_calib, width, label='RMSE-Calib')
    rects2 = ax.bar(x + width/2, rms_stereo, width, label='RMSE-Stereo')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMS-error')
    ax.set_xlabel('Extracted out of {} frames'.format(numberimages))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=(0.7,-0.14), shadow=True, fontsize='large')
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(np.around(height,2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    fig.savefig('lab/calibration/results/Extracted-frames-best result {}.png'.format(title))

    #plt.show()

def plot_planar_deviation(array,method):
    planar_deviation= array[:,:,2]
    mean_frames = []
    for i in range(0, len(planar_deviation)):
        mean_frame = np.mean(np.abs(planar_deviation[i]))
        mean_frames.append([i+1, mean_frame])
    mean_frames = np.asarray(mean_frames)
    totmean_dev = np.mean(np.abs(planar_deviation))


    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(mean_frames[:,0] , mean_frames[:,1], s=5.0, color = "b", label = "Mean Frame")
    ax.grid(True)
    #y_mean_e = [mean_error/len(P_3d_obj), mean_error/len(P_3d_obj)]
    ax.plot(mean_frames[:,0], np.full(len(mean_frames), totmean_dev), color = "r",label = "Total Mean= \u00B1 {} mm".format(np.round_((totmean_dev/1.0),2)), linestyle = "--") #rett linje 
    ax.set(title='Plane Tolerance Deviation {}'.format(method) , ylabel= u"\u00B1 Deviation (mm)",  xlabel='Frames')
    legend = ax.legend(loc=(0.7,-0.14), shadow=True, fontsize='large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#afeeee')

    fig.savefig('lab/calibration/results/Planar-Deviation-{}.png'.format(method))

def main():
    #-----------Initialize all parameters------------#
    
    #Number of inner corners.
    cols = 17 #No. Columns
    rows = 11 #No. Rows

    #Import images and .zdf files from the folders location.
    images = glob.glob("lab/calibration/captured_images/*.png")
    zdf = glob.glob("lab/calibration/zdf/*.zdf")
    #Method, use either: 'corner' or 'center'.
    #'corner' use all corners detected in the frames by findChessBoardCorners() and find the correspondent pixel in the point cloud.
    #'center' find the center of all squares in the frames from findChessBoardCorners() and find the correspondent pixel in the point cloud.
    method = ['corner', 'center']

    
    #Intrinsics parameters of the 3D camera-Zivids camera intrinsics and distortion coeffsients.
    #zivid old one camera
    #camera_intrinsics = np.array([[2767.193359, 0 , 942.942505], [0, 2766.449119, 633.898],[0, 0, 1]])
    #distortion_coeffsients = np.array([[-0.26513, 0.282772, 0.000767328, 0.000367037,0]])
    
    #zivid two camera parameters
    cam_int = np.array([[1782.09204101562, 0, 977.639282226562],
                    [0, 1782.05212402344, 587.777648925781],
                    [0,0,1]])
    cam_dist = np.array([-0.0907880067825317, 0.134410485625267,0.000578985665924847, -5.82622851652559e-05,-0.0652082785964012,])


    #Flags from OpenCV which can give better accuracy.
    flags = cv2.CALIB_ZERO_TANGENT_DIST

    #The projector resoultion width x height (pixels).
    projector_res_width =  1920
    projector_res_height = 1080 

    #The projected size of the pixels in the projected checkerboard image.
    square_size = 60 #120

    #Interpolate method, either: 'nearest' or 'linear'. This is if the pixel return a nan value in the point cloud.
    #'nearest' iterate to the nearest pixels around the set interpolation number.
    #'linear' iterate with bilinaear interpolation around the set interpolation number.
    interpolate_method = 'nearest'
    interpolate_number = 10
    
    #'images_to_iterate'Set a number of frames to use in the calibration.
    #'number_under_len' Number of frames under the total frames. 
    # This is a number so you can pick out a 'number_under_len' frames under the total frames in folder 'images' to iterate over many different combiantions of the frames. 
    #'number_shuffled_iteration'. The number of random shuffled iteration of the number of frames extracted  
    images_to_iterate = len(images)
    number_under_len = 10
    number_shuffled_iteration = 10
    

    #Run ProjextorCalibrate from the Class ProjectorCalibrate
    for i in range(0, len(method)):
        #Calib = ProjectorCalibrate(cols, rows, images, zdf, camera_intrinsics, distortion_coeffsients, projector_res_width, projector_res_height,method[i], images_to_iterate, flags, number_shuffled_iteration, number_under_len, square_size, interpolate_method, interpolate_number)
        Calib = ProjectorCalibrate(cols, rows, images, zdf, cam_int, cam_dist, projector_res_width, projector_res_height,method[i], images_to_iterate, flags, number_shuffled_iteration, number_under_len, square_size, interpolate_method, interpolate_number)


        flag , noflag, framepoints, P_3d_plan, nan_pixels =Calib.ProjectorCalibrate()
        flag = np.asarray(flag)
        noflag = np.asarray(noflag)
        #This return in the following order
        #'flag' and 'noflag' returns all the intrinsics and stereo paremeters with and without opencv flags in the following order.
        #flag[i] - returns the best calibration result in order of the return from CalibrateCamera(). Where flag[0] is the best result.
        #'number_under_len' is equal to the the number of i
        #flag[0][0] - returns the best iteration of the return value of Calibratecamera()
        #flag[0][1] - returns the relative retval from StereoCalibrate to the ret value.
        #flag[0][2] - returns the projector intrinsics matrix
        #flag[0][3] - returns the distortion coefficents of projector 
        #flag[0][4] - return the average reprojection error from the number of frames used in the calibration
        #flag[0][5] - return an array of the reprojection error in each frame.
        #flag[0][6] - return the rotation matrix from the camera to the projector. Given that the camera is orientated left for the projector. 
        #flag[0][7] - return the translation vector from camera to the projector. Same condition as in the previous line.
        #flag[0][8] - return the number of frames used in the folder 'images'
        #flag[0][9] - return the iteration number
        # noflag is constructed in the same way, just that there are set no flags in the calibration.
        # framepoints, are all the checkerboard coordinates in the point cloud detected by the findchessboardCorners()  

        #Visualizing the extrinsics parameters from the camera to the projector
        visualizationstereo(framepoints[0], flag[0][6],flag[0][7], "Extrinsics Parameters with flag {}".format(method[i]))
        visualizationstereo(framepoints[0], noflag[0][6],noflag[0][7], "Extrinsics Parameters without flag{}".format(method[i]))

        plot_reprojection(flag[0][5], flag[0][4],"Neglected tangential distortion, {} ".format(method[i]))
        plot_frame_rms_error(flag[:,8],flag[:,0], flag[:,1], "RMS-error relative to extracted frames, neglected tangential distortion, {}".format(method[i]), images_to_iterate)

        plot_reprojection(noflag[0][5], noflag[0][4],"{}".format(method[i]))
        plot_frame_rms_error(noflag[:,8],noflag[:,0], noflag[:,1], "RMS-error relative to extracted frames, {}".format(method[i]), images_to_iterate)
        
        plot_planar_deviation(P_3d_plan, method[i])
        print("Method: ", method[i], "with OpenCV Flag","number of frames:", flag[0][8], "/", images_to_iterate, "iteration number:", flag[0][9], "/", number_shuffled_iteration, ".")
        print("return value from calibrateCamera() : ", flag[0][0])
        print("\n return value from stereoCalibrate() : ", flag[0][1])
        print("\ Projector intrinsics: \n", flag[0][2])
        print("\n Projector distortion coefficents:\n",flag[0][3])
        print("\nThe rotation vector between camera and projector:\n",flag[0][6])
        print("\nThe translationg vector between camera and projector:\n",flag[0][7])

        print("Method: ", method[i], "with NO OpenCV Flag","number of frames:", noflag[0][8], "/", images_to_iterate, "iteration number:", noflag[0][9], "/", number_shuffled_iteration, ".")
        print("return value from calibrateCamera() : ", noflag[0][0])
        print("\n return value from stereoCalibrate() : ", noflag[0][1])
        print("\ Projector intrinsics: \n", noflag[0][2])
        print("\n Projector distortion coefficents:\n",noflag[0][3])
        print("\nThe rotation vector between camera and projector:\n",noflag[0][6])
        print("\nThe translationg vector between camera and projector:\n",noflag[0][7])
        for j in range(len(flag)):
            fs = cv2.FileStorage(f'lab\\calibration\\csv\\{method[i]}\\flag\{j}-parameters.csv', cv2.FILE_STORAGE_WRITE)
            fs.write('rms_camera', flag[j][0])
            fs.write('rms_stereo', flag[j][1])
            fs.write('proj_int', flag[j][2])
            fs.write('proj_dist', flag[j][3])
            fs.write('avg_error', flag[j][4])
            #fs.write('frame_error', flag[j][5])
            fs.write('rotation', flag[j][6])
            fs.write('translation', flag[j][7])
            fs.write('no_frames', flag[j][8])
            fs.write('it_no', flag[j][9])
            fs.release()
            fk =  cv2.FileStorage(f'lab\\calibration\\csv\\{method[i]}\\noflag\{j}-parameters.csv', cv2.FILE_STORAGE_WRITE)
            fk.write('rms_camera', noflag[j][0])
            fk.write('rms_stereo', noflag[j][1])
            fk.write('proj_int', noflag[j][2])
            fk.write('proj_dist', noflag[j][3])
            fk.write('avg_error', noflag[j][4])
            #fs.write('frame_error', noflag[j][5])
            fk.write('rotation', noflag[j][6])
            fk.write('translation', noflag[j][7])
            fk.write('no_frames', noflag[j][8])
            fk.write('it_no', noflag[j][9])
            fk.release()

    print("done")
if __name__ == "__main__":
    main()