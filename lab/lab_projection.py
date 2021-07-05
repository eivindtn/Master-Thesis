#MIT License

#Copyright (c) 2021 Eivind Torsrud Nerol

#pip package dependices
import numpy as np
import open3d as o3d
import cv2
import vedo as v
import zivid

#functions and settings 
import geometry_vision as gv
import capture as ca
import settings as se

#load in the files or capture and visualize in the lab area
states = ['lab', 'load']
state = states[1]

#modes of testing for either demo, evaluation and accuracy 
modes = ['section_projection', 'center_projection', 'evaluation']
mode = modes[0]
no_experiment = 12
no_calibration = 0
center = 'show_not'
toe_mode = False
figure_plotter =  True

#paths to files
settings_file = 'lab\\settings\\lab_settings.yml' #capture settings for the zivid two camera exported from zivid studio
output_path = 'lab/result/Experiment_' + f'{no_experiment}' + '\\leg.zdf' #path to zivid data file format
ply_path = 'lab/result/Experiment_' + f'{no_experiment}' + '\\leg.ply' #path to ply format so the data can be used into other python packages
proj_calib_path =f'lab\\calibration\\csv\\center\\noflag\{no_calibration}-parameters.csv' #path to the intrinsics and extrinsics parameters of the projector-zivid system

#path to experiment files
result_ply_path = 'lab/result/Experiment_' + f'{no_experiment}' + '/config_'
result_output_path = 'lab/result/Experiment_' + f'{no_experiment}' + '/config_'

#load intrinsics, extrinsics parameters for projector-zivid setup
cam_int, cam_dist, cam_res = se.zivid_parameters()
proj_intr, proj_dist, t_ext, proj_res = se.proj_calib_parameter(proj_calib_path)
SCREEN_ID = 0 #projector ID
#viz = o3d.visualization.draw_geometries #visualization through open3d

#configuration of the stub on the leg within rotation and translation 
radius = 0.105 *1000 # stub radius in mm
thickness = 20      # stub thickness to thicken the section cut onto the leg
length = 1 * 400   # stub length in mm

#arrays for saving mesh, coordinates and result for visualization 
config_result = []
config_result_2 = []
plot = []

#data/noise cleaning parameters
bound_x, bound_y, bound_z = [20,20], [20,20], [100,100] # delete points in the pointcloud outside of the 4 corners of the aruco marker 
tol_clean_fit = 0.033 # cleaning tol for fitting the cylinder points to an axis, radius and center
tol_clean = 0.009 # downsampling cleaning for creating a mesh of the cylinder
z_min = 300 # delete all points within less than z_min from the zivid camera
z_max = 800 # same as above but a max parameter

#capture a point cloud from the zivid camera
if state == 'lab':
    leg_point_cloud, xyz, rgba = ca.capture_w_yaml(settings_file, output_path, ply_path)
    init_image= v.Picture(rgba[:, :, 0:3])
    init_image.write('lab/results/Experiment_' + f'{no_experiment}'+'/leg.png')
else:
    leg_point_cloud, xyz, rgba = ca.load_zdf_frames(output_path, ply_path)
#visualization of the point cloud
#v.show(leg_point_cloud, v.Picture(rgba[:, :, 0:3]),N=2 , sharecam = False, axes =3).close()
#v.show([[leg_point_cloud],[],[]],N=3, sharecam = False)
#leg_c = o3d.io.read_point_cloud("lab/results/chord-1.ply")
#leg_point_cloud = v.load("lab/results/chord-1.ply")

aruco_point, corners_cloud = gv.find_aruco_point(cv2.aruco.DICT_4X4_50, xyz, 4, rgba)

#downsampling, cleaning and delete points from the captured point cloud
leg_point_cloud_clone, aruco_marker_cloud = gv.delete_aruco_marker_from_point_cloud(leg_point_cloud, corners_cloud, bound_x, bound_y, bound_z)
leg_point_cloud_clone.deletePoints(np.where(leg_point_cloud.points()[:,2] >= z_max)[0])
leg_point_cloud_clone.deletePoints(np.where(leg_point_cloud.points()[:,2] <= z_min)[0])
leg_fit_cloud = leg_point_cloud_clone.clone().clean(tol=tol_clean_fit)
leg_point_cloud_clone.clean(tol=tol_clean)

#leg_c_d = leg_c.voxel_down_sample(voxel_size=10) # downsample pointcloud
leg_axis, leg_center, leg_r, fit_error = gv.fit(np.asarray(leg_fit_cloud.points())) #cylinder fitting to the captured pointcloud
#leg_axis = abs(leg_axis)
stub_axis = gv.perpendicular_vector(leg_axis)
axis_p1 = aruco_point+stub_axis*300 # define two points along the stub axis 
axis_p2 = aruco_point-stub_axis*300 

#define a configuration of the stub onto the leg
axes = np.array([gv.perpendicular_vector(stub_axis), leg_axis, stub_axis]) #refrence systemn for x,y,z axis for transformation of the stub
rotation = np.array([[0,0,0],
                     [0,0,0],
                     [0,0,0]]) #rotation in degrees around the axes 
                     
translation = np.array([leg_axis*200,     #alternative configure the translation manually
                        leg_axis*300,
                        leg_axis*400]) #translation along the axes defined two lins above
#find a way to draw these axes either by open3d or vedo
plane_axes = np.cross(stub_axis, leg_axis)
plane_axes_2 =np.cross(axes[0], axes[2]) 
#create a leg_mesh
leg_mesh = v.pointcloud.delaunay2D(leg_point_cloud_clone.points(), mode='fit').subdivide(3) # create leg mesh from blender point cloud
leg_mesh = leg_mesh.triangulate().alpha(0.2)
intersect_c = leg_mesh.intersectWithLine(axis_p1, axis_p2)

if figure_plotter:
    leg_p1 = leg_center +300*leg_axis
    leg_p2 = leg_center -300*leg_axis
    axis = v.shapes.Line(leg_p1,leg_p2)
    axis_2 = v.shapes.Line(axis_p1,axis_p2)
    ref_y = v.shapes.Arrow(intersect_c[0],np.array(intersect_c+axes[0]*100)[0]).c('g')
    ref_x = v.shapes.Arrow(intersect_c[0],np.array(intersect_c+axes[1]*100)[0]).c('r')
    ref_z = v.shapes.Arrow(intersect_c[0],np.array(intersect_c+axes[2]*-100)[0]).c('b')
    ref = v.Assembly(ref_x, ref_y, ref_z)
    test_cyl =v.shapes.Cylinder(pos=leg_center, r=leg_r, height=2000, axis=leg_axis, c='teal3', alpha=1, cap=False, res=800)
    #create a self made coordinate system
    x= v.shapes.Arrow([0,0,0],[200,0,0]).c('r')
    y= v.shapes.Arrow([0,0,0],[0,200,0]).c('g')
    z= v.shapes.Arrow([0,0,0],[0,0,200]).c('b')
    cam_sys = v.Assembly(x,y,z)
    x_p = x.clone().applyTransform(t_ext)
    y_p = y.clone().applyTransform(t_ext)
    z_p = z.clone().applyTransform(t_ext)
    proj_syst = v.Assembly(x_p,y_p,z_p)
    #v.show(leg_fit_cloud.c('r').ps(3),leg_point_cloud_clone,test_cyl.c('o'), axis.c('b').lw(5)).close()

#synthetic stub generation
stub  = v.shapes.Cylinder(pos=intersect_c[0], r=radius, height=length, axis=stub_axis, c='teal3', alpha=1, cap=False, res=800)
stub_concentric  = v.shapes.Cylinder(pos=intersect_c[0], r=radius+thickness, height=length, axis=stub_axis, c='teal3', alpha=1, cap=False, res=800)
stub_mesh = stub.triangulate().alpha(0.2)
stub_concentric_mesh = stub_concentric.triangulate().alpha(0.2)
#add a concentric cylinder which can define a more accurate section-cut 

for c in range (len(translation)):
    blank_image = np.zeros((1080,1920,3), np.uint8)
    if mode == 'section_projection' or mode == 'center_projection':
        stub_config = stub_mesh.clone()
        stub_concentric_config = stub_concentric_mesh.clone()
        T = np.identity(4)
        T[:3,3] = translation[c]
        for i in range(len(axes)):
            stub_config.rotate(rotation[c][i],axis=axes[i],point=intersect_c[0])
            stub_concentric_config.rotate(rotation[c][i],axis=axes[i],point=intersect_c[0])
        stub_config.applyTransform(T)
        stub_concentric_config.applyTransform(T)

        intersect_curve = leg_mesh.intersectWith(stub_config).join(reset=True) # fix the boundary in the array and intersect two mesh
        intersect_concentric_curve = leg_mesh.intersectWith(stub_concentric_config).join(reset=True)
        spline3d = v.shapes.Spline(intersect_curve)
        spline3d_concentric = v.shapes.Spline(intersect_concentric_curve)
        grind_surf = leg_mesh.clone()
        grind_surf.cutWithMesh(stub_concentric_config, invert = False) #Alternative use cutWithCylinder which much faster
        grind_surf.cutWithMesh(stub_config, invert = True)
    if mode == 'center_projection':

        #compute the greatest distance from the centerpoint of the stub to the curve
        center3d = (intersect_c[0]+translation[c])
        plane_ax1 = v.shapes.Plane(pos=(center3d[0],center3d[1], center3d[2]), normal=(axes[0][0], axes[0][1], axes[0][2]), sx=1000, sy=1000, c='gray6', alpha=1)
        plane_ax1.triangulate()
        #v.show(v.Points(l), v.Points([center2d]), line2d, leg_mesh, intersect_curve, Ribbon, aruco_marker_cloud,stub_config, plane_ax1).close()
        '''
        center2d = (intersect_c[0]+translation[c])[0:2]
        l = list(intersect_curve.points()[:,0:2])
        l.sort(key=lambda coord: gv.dist(center2d[0], center2d[1] , coord[0], coord[1]))
        sorted_arr = np.array(l)
        direct = (-sorted_arr[-1]+center2d[0:2])
        a1 = sorted_arr[-1] -10*direct
        a2 = center2d[0:2]+10*direct
        line2d  = v.shapes.Line(a1,a2)

        a2_3d = np.append(a2, 2*1000)
        a1_3d = np.append(a1,2*1000)
        a1 = np.append(a1,0)
        a2 = np.append(a2,0)

        line3d = v.shapes.Line(a1_3d, a2_3d)
        Ribbon = v.shapes.Ribbon(line2d, line3d)
        Ribbon.triangulate()
        r = v.shapes.Line(a1, a1_3d)'''
        center_line = leg_mesh.intersectWith(plane_ax1).join(reset=True) # Ribbbon for greatest distance
 
        hits = []
        for i in center_line.points():
            cpt = spline3d_concentric.closestPoint(i)
            if v.mag2(i-cpt)<0.2:
                hits.append(i)
        # print(array(hits))
        hits = v.Points(hits, r=10, c='r')

        toe_t  = v.shapes.Text3D('X', pos=hits.points()[0] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
        heel_t  = v.shapes.Text3D('X', pos=hits.points()[-1] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
        center_t  = v.shapes.Text3D('X', pos=intersect_c[0]+translation[c] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)

        # create an projected image
        
        centerpoints = center_t.clone()
        toe = toe_t.clone()
        heel = heel_t.clone()
        centerpoints.subdivide(5, method=1)
        toe.subdivide(5, method=1)
        heel.subdivide(5, method=1)
        centerpoints_2d = gv.point_to_xy(centerpoints.clone().applyTransform(t_ext).points(),proj_intr)
        toe_2d = gv.point_to_xy(toe.clone().applyTransform(t_ext).points(),proj_intr)
        heel_2d = gv.point_to_xy(heel.clone().applyTransform(t_ext).points(),proj_intr)
        centerline_points_2d = gv.point_to_xy(center_line.clone().applyTransform(t_ext).points(),proj_intr)

        for i in range (0, len(centerpoints_2d)):
            blank_image[centerpoints_2d[i][1]][centerpoints_2d[i][0]]= [0,0,255]

        for i in range (0, len(toe_2d)):
            blank_image[toe_2d[i][1]][toe_2d[i][0]]= [0,0,255]

        for i in range (0, len(toe_2d)):
            blank_image[heel_2d[i][1]][heel_2d[i][0]]= [0,0,255]


    if mode == 'section_projection':

        splinepoints_2d = gv.point_to_xy(spline3d.clone().applyTransform(t_ext).points(),proj_intr)
        splinepoints_concentric_2d = gv.point_to_xy(spline3d_concentric.clone().applyTransform(t_ext).points(),proj_intr)
        
        img =blank_image.copy()
        img2 = cv2.fillPoly(img, [splinepoints_concentric_2d], [0,255,0])
        blank_image = cv2.fillPoly(img2, [splinepoints_2d], [0,0,0])
        for i in range (0, len(splinepoints_2d)):
            blank_image[splinepoints_2d[i][1]][splinepoints_2d[i][0]]= [0,255,0]
        
        for i in range (0, len(splinepoints_concentric_2d)):
            blank_image[splinepoints_concentric_2d[i][1]][splinepoints_concentric_2d[i][0]]= [0,255,0]
        if center == 'show':
            center_t  = v.shapes.Text3D('X', pos=intersect_c[0]+translation[c] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
            centerpoints = center_t.clone()
            centerpoints.subdivide(5, method=1)
            centerpoints_2d = gv.point_to_xy(centerpoints.clone().applyTransform(t_ext).points(),proj_intr)
            for i in range (0, len(centerpoints_2d)):
                blank_image[centerpoints_2d[i][1]][centerpoints_2d[i][0]]= [225,255,0]
    if toe_mode:

        #compute the greatest distance from the centerpoint of the stub to the curve
        center3d = (intersect_c[0]+translation[c])
        plane_ax1 = v.shapes.Plane(pos=(center3d[0],center3d[1], center3d[2]), normal=(plane_axes_2[0], plane_axes_2[1], plane_axes_2[2]), sx=1000, sy=1000, c='gray6', alpha=1)
        plane_ax1.triangulate()
        #v.show(v.Points(l), v.Points([center2d]), line2d, leg_mesh, intersect_curve, Ribbon, aruco_marker_cloud,stub_config, plane_ax1).close()
        '''
        center2d = (intersect_c[0]+translation[c])[0:2]
        l = list(intersect_curve.points()[:,0:2])
        l.sort(key=lambda coord: gv.dist(center2d[0], center2d[1] , coord[0], coord[1]))
        sorted_arr = np.array(l)
        direct = (-sorted_arr[-1]+center2d[0:2])
        a1 = sorted_arr[-1] -10*direct
        a2 = center2d[0:2]+10*direct
        line2d  = v.shapes.Line(a1,a2)

        a2_3d = np.append(a2, 2*1000)
        a1_3d = np.append(a1,2*1000)
        a1 = np.append(a1,0)
        a2 = np.append(a2,0)

        line3d = v.shapes.Line(a1_3d, a2_3d)
        Ribbon = v.shapes.Ribbon(line2d, line3d)
        Ribbon.triangulate()
        r = v.shapes.Line(a1, a1_3d)'''
        center_line = leg_mesh.intersectWith(plane_ax1).join(reset=True) # Ribbbon for greatest distance
 
        hits = []
        for i in center_line.points():
            cpt = spline3d_concentric.closestPoint(i)
            if v.mag2(i-cpt)<0.2:
                hits.append(i)
        # print(array(hits))
        hits = v.Points(hits, r=10, c='r')

        toe_t  = v.shapes.Text3D('X', pos=hits.points()[0] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
        heel_t  = v.shapes.Text3D('X', pos=hits.points()[-1] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
        center_t  = v.shapes.Text3D('X', pos=intersect_c[0]+translation[c] , s=30, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)

        # create an projected image
        center_3d = v.Points([intersect_c[0]+translation[c]])
        #centerpoints = center_t.clone()
        #toe = toe_t.clone()
        #heel = heel_t.clone()
        #centerpoints.subdivide(5, method=1)
        #toe.subdivide(5, method=1)
        #heel.subdivide(5, method=1)
        #centerpoints_2d = gv.point_to_xy(centerpoints.clone().applyTransform(t_ext).points(),proj_intr)
        #toe_2d = gv.point_to_xy(toe.clone().applyTransform(t_ext).points(),proj_intr)
        #heel_2d = gv.point_to_xy(heel.clone().applyTransform(t_ext).points(),proj_intr)
        #centerline_points_2d = gv.point_to_xy(center_line.clone().applyTransform(t_ext).points(),proj_intr)
        cen=intersect_c[0]+translation[c]
        center_3d  = v.Points([cen])
        center_2d =  gv.point_to_xy(center_3d.clone().applyTransform(t_ext).points(),proj_intr)
        toe_heel= gv.point_to_xy(hits.clone().applyTransform(t_ext).points(),proj_intr)
        for p in toe_heel:
            im_with_keypoints = cv2.drawMarker(blank_image,
                                           (p[0],p[1]),
                                           (0, 0, 255),
                                           markerType=cv2.MARKER_CROSS,
                                           markerSize=100,
                                           thickness=10,
                                           line_type=cv2.LINE_AA)
        im_with_keypoints = cv2.drawMarker(blank_image,
                                           (center_2d[0][0],center_2d[0][1]),
                                           (0, 0, 255),
                                           markerType=cv2.MARKER_CROSS,
                                           markerSize=100,
                                           thickness=10,
                                           line_type=cv2.LINE_AA)
        #for i in range (0, len(centerpoints_2d)):
            #blank_image[centerpoints_2d[i][1]][centerpoints_2d[i][0]]= [0,0,255]

        #for i in range (0, len(toe_2d)):
            #blank_image[toe_2d[i][1]][toe_2d[i][0]]= [0,0,255]

        #for i in range (0, len(toe_2d)):
            #blank_image[heel_2d[i][1]][heel_2d[i][0]]= [0,0,255]  



    #for i in range (0, len(centerline_points_2d)):
        #blank_image[centerline_points_2d[i][1]][centerline_points_2d[i][0]]= [225,128,255]
    
    '''gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for ci in cnts:
        cv2.drawContours(blank_image, [ci], -1, (255,128,255), thickness=5)'''
    
    #for i in range (0, len(centerpoints_2d)):
        #blank_image[centerpoints_2d[i][1]][centerpoints_2d[i][0]]= [225,255,0]

    #for i in range (0, len(toe_2d)):
        #blank_image[toe_2d[i][1]][toe_2d[i][0]]= [0,0,255]

    #for i in range (0, len(toe_2d)):
        #blank_image[heel_2d[i][1]][heel_2d[i][0]]= [0,255,0]

    #This is the distorted image
    pic_dst =v.Picture(blank_image)

    #Now we need to undistort the image
    # undistort
    h,  w = blank_image.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(proj_intr,proj_dist,(w,h),1,(w,h))
    undst = cv2.undistort(blank_image, proj_intr, proj_dist, None, proj_intr)

    pic_undst = v.Picture(undst)
    if state == 'lab':
        PROJ_WIN = "projector_win"
        cv2.namedWindow(PROJ_WIN, cv2.WND_PROP_FULLSCREEN)
        proj_w, proj_h = ca.init_proj(PROJ_WIN, SCREEN_ID)
        cv2.imshow(PROJ_WIN, blank_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        leg_point_cloud_projection, xyz_projection, rgba_projection = ca.capture_w_yaml(settings_file, result_output_path+f'{c}\\after_projection.zdf', result_ply_path+f'{c}\\after_projection.ply')
    else:
        leg_point_cloud_projection, xyz_projection, rgba_projection = ca.load_zdf_frames(result_output_path+f'{c}\\after_projection.zdf', result_ply_path+f'{c}\\after_projection.ply')

    if mode == 'section_projection':
        #section_cut_points, thresh = gv.find_image_structural_simularity(rgba[:, :, 0:3], rgba_projection[:, :, 0:3], xyz, show= True)
        section_cut_points, thresh = gv.morphing(rgba[:, :, 0:3], rgba_projection[:, :, 0:3],xyz_projection, np.ones((5,5),np.uint8), show= False)
        section_cut_points = v.Points(section_cut_points).clean(tol=0.001)
        section_cut_points.deletePoints(np.where(section_cut_points.points()[:,2] >= z_max)[0]) #added cleaning by Z-componen
        section_cut_points.deletePoints(np.where(section_cut_points.points()[:,2] <= z_min)[0]) 
        print(len(section_cut_points.points()))
        section_points = []
        for p in section_cut_points.points():
            cpt = spline3d.closestPoint(p)
            if v.mag2(p-cpt)<1000:
                section_points.append(p)
        projected_section = v.removeOutliers(section_points, radius = 8)
        projected_section_p = v.Points(projected_section)
        projected_section_p.alignTo(grind_surf, rigid = True)
        grind_surf_projected  = v.recoSurface(projected_section, radius= 3, dims=(1000,1000,500))
        align_transformation = np.eye(4)
        projected_section_p.GetMatrix().DeepCopy(align_transformation.ravel(), projected_section_p.GetMatrix())
        grind_surf_projected_signed = grind_surf_projected.clone()
        grind_surf_projected_signed.distanceToMesh(grind_surf, signed=True, negate=False)
        grind_surf_projected_signed.addScalarBar(title='Signed\nDistance')
        signed_dist = grind_surf_projected_signed.getPointArray("Distance")
        signed_dist_avg = np.average(signed_dist)
        #projection_mesh = v.recoSurface(projected_section, dims=200, radius=5)
        #original_projection_mesh = leg_mesh.clone()
        #original_projection_mesh.cutWithLine(intersect_concentric_curve.points())
        #original_projection_mesh.cutWithLine(intersect_curve.points())
        
        '''
        d
        inner_spline = []
        for p in projected_section:
            cpt = spline3d.closestPoint(p)
            if v.mag2(p-cpt)<100:
                inner_spline.append(p)
        inner_spline = v.Points(inner_spline, r=5)
        N=4
        for i in range(1, N):
            inner_spline = inner_spline.clone().smoothMLS1D(f=0.4).color(i)

            if i == N-1:
                # at the last iteration make sure points
                # are separated by tol (in % of bbox)
                inner_spline.clean(tol=0.02)

        outer_spline = []
        for p in projected_section:
            cpt = spline3d_concentric.closestPoint(p)
            if v.mag2(p-cpt)<100:
                outer_spline.append(p)
        
        outer_spline = v.Points(outer_spline, r=5)
        for i in range(1, N):
            outer_spline = outer_spline.clone().smoothMLS1D(f=0.4).color(i)

            if i == N-1:
                # at the last iteration make sure points
                # are separated by tol (in % of bbox)
                outer_spline.clean(tol=0.02)
        
        distance_inner = 0
        t_inner = []
        for p in inner_spline.points():
            cpt = spline3d.closestPoint(p)
            l =v.mag2(p - cpt)
            distance_inner += l  # square of residual distance
            t_inner.append(l)
            print(l)

        print("ave. squared distance =", distance_inner/inner_spline.N())

        distance_outer = 0
        t_outer = []
        for p in outer_spline.points():
            cpt = spline3d_concentric.closestPoint(p)
            m =v.mag2(p - cpt)
            distance_outer += m  # square of residual distance
            t_outer.append(m)
            print(m)

        print("ave. squared distance =", distance_outer/outer_spline.N())
        '''

        #testpoints = v.removeOutliers(section_cut_points, radius=4)
        #testpoints = v.Points(testpoints).clean(tol=0.01)
        '''inner_spline = []
        for p in testpoints.points():
            cpt = spline3d.closestPoint(p)
            if v.mag2(p-cpt)<80:
                inner_spline.append(p)

        outer_spline = []
        for p in testpoints.points():
            cpt = spline3d_concentric.closestPoint(p)
            if v.mag2(p-cpt)<80:
                outer_spline.append(p)
        inner_spline = v.Points(inner_spline).clean(0.03)
        outer_spline = v.Points(outer_spline).clean(0.03)'''
        config_result_2.append([leg_mesh,stub_config, stub_concentric_config, intersect_curve, intersect_concentric_curve, spline3d, spline3d_concentric, aruco_marker_cloud, v.Points(intersect_c), v.Points([intersect_c[0]+translation[c]])])
        plot.append([grind_surf, grind_surf_projected, projected_section, projected_section_p, grind_surf_projected_signed, align_transformation, signed_dist, signed_dist_avg, thresh,leg_point_cloud_projection, blank_image])
        #config_result.append([stub_config, stub_concentric_config, intersect_curve, intersect_concentric_curve, spline3d, spline3d_concentric, inner_spline, outer_spline, aruco_marker_cloud, v.Points(intersect_c), v.Points([intersect_c[0]+translation[c]])])
        #v.show(leg_mesh,config_result[c][0],config_result[c][1],config_result[c][9].c('r').ps(5), config_result[c][10].c('r').ps(5), config_result[c][8], axes=3).export(result_output_path+f"{c}/config.npz").close()
        #v.show(leg_mesh,config_result[c][4].c('r'), config_result[c][5].c('r'), config_result[c][6].c('g'), config_result[c][7].c('g'), axes=3).export(result_output_path+f"{c}/section_line_comparison.npz").close()
        
        thresh = v.Picture(thresh)
        pic = v.Picture(rgba_projection[:, :, 0:3])
        pic.write(result_output_path+f"{c}/img.png")
        pic_dst.write(result_output_path+f"{c}/projector_img.png")
        thresh.write(result_output_path+f"{c}/thresh.png")

    print('done')

#config figure
v.show([[config_result_2[0][0],config_result_2[0][1],config_result_2[0][2],plot[0][0].c('black'),config_result_2[0][4].c('b').lw(5),config_result_2[0][5].c('b').lw(5), ref,cam_sys, proj_syst],[config_result_2[1][0],config_result_2[1][1],config_result_2[1][2],plot[1][0].c('black'),config_result_2[1][4].c('b').lw(5),config_result_2[1][5].c('b').lw(5), ref,cam_sys, proj_syst],[config_result_2[2][0],config_result_2[2][1],config_result_2[2][2],plot[2][0].c('black'),config_result_2[2][4].c('b').lw(5),config_result_2[2][5].c('b').lw(5), ref,cam_sys, proj_syst]], N=3).close()

#projector projected image
v.show(v.Picture(plot[0][-1]),v.Picture(plot[1][-1]),v.Picture(plot[2][-1]), N=3, axes=1).close()

#thresh plot
v.show(v.Picture(plot[0][-3]),v.Picture(plot[1][-3]),v.Picture(plot[2][-3]), N=3, axes=1).close()

#all sections 
v.show(plot[0][0].c('g').ps(5),v.Points(plot[0][2]).c('r'), plot[1][0].c('g').ps(5),v.Points(plot[1][2]).c('r'),plot[2][0].c('g').ps(5),v.Points(plot[2][2]).c('r'), leg_point_cloud_projection).close()

#show point deviation 
v.show([[plot[0][0].c('black').ps(5),v.Points(plot[0][2]).c('r'), leg_point_cloud_projection, cam_sys],[plot[1][0].c('black').ps(5),v.Points(plot[1][2]).c('r'), leg_point_cloud_projection, cam_sys],
[plot[2][0].c('black').ps(5),v.Points(plot[2][2]).c('r'), leg_point_cloud_projection, cam_sys],[plot[0][0].c('black').ps(5),v.Points(plot[0][2]).c('r')],[plot[1][0].c('black').ps(5),v.Points(plot[1][2]).c('r')],
[plot[2][0].c('black').ps(5),v.Points(plot[2][2]).c('r')]], N=6).close()

#Signed distance
v.show([[plot[0][0], plot[0][4]],[plot[1][0], plot[1][4]],[plot[2][0], plot[2][4]]], N=3).close()

#Alligned after ICP
v.show([[plot[0][0], plot[0][3].c('r')],[plot[1][0], plot[1][3].c('r')],[plot[2][0], plot[2][3].c('r')]], N=3).close()

#print transformation matrix
for t in range(3):
    trans = plot[t][5]
    rot = plot[t][5][:3,:3]
    rot_e = gv.R_to_eul(rot)
    rot_e_d = rot_e.as_euler('xyz', degrees = True)
    translat = plot[t][5][:3,3]
    norm = np.linalg.norm(translat)
    print("Number", t)
    print("Transformation\n",trans)
    print("Rotation euler xyz",rot_e_d)
    print("Norm translat", norm)
    print("Average signed dist", plot[t][7])
    print("initial surface area ", plot[t][0].area())
    print("projected surface area ", plot[t][1].area())

#histogram
hst1 = v.pyplot.histogram(plot[0][6],
                 bins=30,
                 errors=True,
                 aspect=4/3,
                 title='',
                 xtitle='Signed distance (mm)',
		 ytitle = 'Number of points',
                 c='red',
                 marker='o',
                )
hst2 = v.pyplot.histogram(plot[1][6],
                 bins=30,
                 errors=True,
                 aspect=4/3,
                 title='',
                 xtitle='Signed distance (mm)',
		 ytitle = 'Number of points',
                 c='blue',
                 marker='o',
                )
hst3 = v.pyplot.histogram(plot[2][6],
                 bins=30,
                 errors=True,
                 aspect=4/3,
                 title='',
                 xtitle='Signed distance (mm)',
		 ytitle = 'Number of points',
                 c='green',
                 marker='o',
                )
v.show(hst1, hst2, hst3, N=3).close()
#Plott the result
#v.show([[leg_mesh,config_result[0][4].c('r'), config_result[0][5].c('r'), inner_spline.c('g'), outer_spline.c('g')], [leg_point_cloud_projection, inner_spline.c('y').ps(5), outer_spline.c('y').ps(5)] ,[v.Picture(rgba_projection[:,:,0:3])]],N=3, sharecam= False,axes=3).close()
#v.show([[leg_mesh,config_result[0][4].c('r'), config_result[0][5].c('r'), inner_spline.c('g'), outer_spline.c('g')], [leg_point_cloud_projection, inner_spline.c('y').ps(5), outer_spline.c('y').ps(5)]],N=2, sharecam= False,axes=3).export("scene.npz").close()

#config_result[c][6].cmap('rainbow', t_inner).addScalarBar3D(title='\pm1/\sqrtR')

#v.show([[pic_dst], [pic]]) mm
#v.show([[leg_mesh, stub_config, spline3d, spline3d_concentric, center_t],[leg_point_cloud_projection,spline3d, spline3d_concentric],
#         [pic_dst,pic_undst]], N=3).close()

'''cv2.imwrite('projection-4.png', blank_image)
v.show([
        [leg_mesh, stub_config, toe_t, heel_t, center_t, center_line,intersect_curve],
        [leg_point_cloud_2, v.Points([aruco_point]).c('r'), stub_config],[pic]                            # second renderer
      ], N=3, sharecam=False).close()
xyz, rgba = gv.load_zdf_frames('lab/results\\chord-3-section-projected.zdf', 'lab/results\\chord-3-section-projected.ply', visualization= False)
  https://github.com/zivid/zivid-python-samples/blob/master/source/camera/basic/capture_hdr_complete_settings.py
a  https://github.com/zivid/zivid-python-samples/blob/master/source/camera/basic/capture_with_settings_from_yml.py

while(True):
    PROJ_WIN = "projector_win"
    SCREEN_ID = 0
    cv2.namedWindow(PROJ_WIN, cv2.WND_PROP_FULLSCREEN)
    proj_w, proj_h = ca.init_proj(PROJ_WIN, SCREEN_ID)
    cv2.imshow(PROJ_WIN, undst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''

print('done')
