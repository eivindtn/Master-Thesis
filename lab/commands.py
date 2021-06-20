#define a configuration of the stub onto the leg
axes = np.array([gv.perpendicular_vector(stub_axis), leg_axis, stub_axis]) #refrence systemn for x,y,z axis for transformation of the stub
#create different configurations for different stub assembly
rotation = np.array([[0,0,0],
                     [45,0,40],
                     [30,10,30],    #rotation around the axes 
                     [30,10,30],
                     [30,10,30],
                     [30,10,30]]) #rotation in degrees around the axes 
translation = np.array([[leg_axis*200],     #alternative configure the translation manually
                        [leg_axis*200],
                        [leg_axis*200],     #translation along the  center axis to the cylinder
                        [leg_axis*200],
                        [leg_axis*200],
                        [leg_axis*200]]) #translation along the axes defined two lins above

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco, corners = gv.read_aruco_mark(rgba[:, :, 0:3], dictionary)
aruco_point = xyz[int(aruco[1]),int(aruco[0])]
aruco_point = xyz[np.round(aruco[1]+1,0).astype('int32'),np.round(aruco[0],0).astype('int32')]
corners_cloud = []
for i in range (len(corners[0][0])):
    corners_cloud.append(xyz[int(corners[0][0][i][1])][int(corners[0][0][i][0])]) 
corners_cloud = np.asarray(corners_cloud)

T = np.identity(4)
T[:3,3] = translation


bound_box = [np.min(corners_cloud[:,0])-20,np.max(corners_cloud[:,0])+20,np.min(corners_cloud[:,1])-20,np.max(corners_cloud[:,1])+20, np.min(corners_cloud[:,2])-100,np.max(corners_cloud[:,2])+100]
leg_point_cloud_clone = leg_point_cloud.clone()
aruco_marker_cloud = leg_point_cloud.clone().crop(bounds= bound_box)
hits = []
for p in range (len(aruco_marker_cloud.points())):
    hits.append(leg_point_cloud.closestPoint(aruco_marker_cloud.points()[p],N=1, returnPointId=True))
'''for i in range(len(leg_point_cloud_clone.points())):
    if leg_point_cloud_clone.points()[i,2] >= 1300 or leg_point_cloud_clone.points()[i,2] <= 300 :
        hits.append(i)'''
leg_point_cloud_clone.deletePoints(hits)