import numpy as np
import open3d as o3d
import cv2
import vedo as v
import geometry_vision as gv
from scipy.ndimage import gaussian_filter1d

#load intrinsics, extrinsics, pointcloud from the Blender setup
cam_int = np.genfromtxt("blender/scenes/csv/K_camera.csv", delimiter=",") # camera intrinsics
cam_res = [1920, 1200]
proj_intr = np.genfromtxt("blender/scenes/csv/K_projector.csv", delimiter=",") #projector intrinsics
proj_res = [1920, 1200]
t_ext = np.genfromtxt("blender/scenes/csv/T_1_2.csv", delimiter = ",") # Transformation from cam to proj

intr = o3d.open3d.camera.PinholeCameraIntrinsic(cam_res[1], cam_res[0], fx=cam_int[0][0], fy=cam_int[1][1], cx=cam_int[0][2], cy=cam_int[1][2]) #intrinsic parameters in open3d
leg_c, img, z = gv.pointcloud_blender('blender/scenes/exr/pointcloud.exr', cam_res[0], cam_res[1],intr, mode="simple") #read rendered pointcloud from Blender
leg_c_d = leg_c.voxel_down_sample(voxel_size=0.2) # downsample pointcloud
leg_axis, leg_center, leg_r, fit_error = gv.fit(np.asarray(leg_c_d.points)) #cylinder fitting to the captured pointcloud

o3d.io.write_point_cloud("blender/leg_c.ply", leg_c)
leg_point_cloud = v.load("blender/leg_c.ply")

##add aruco mark to have a reference point
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco, corners = gv.read_aruco_mark(img, dictionary)
aruco_point = gv.pixel_to_point(z[int(aruco[0])][int(aruco[1])],aruco, cam_int) 
corners_cloud = []
for i in range (len(corners[0][0])):
    corners_cloud.append(gv.pixel_to_point(z[int(corners[0][0][i][0])][int(corners[0][0][i][1])],corners[0][0][i], cam_int)) 
corners_cloud = np.asarray(corners_cloud)

#datacleaning, deleting the aruco mark from the pointcloud
bound_box = [np.min(corners_cloud[:,0]),np.max(corners_cloud[:,0])+0.05,np.min(corners_cloud[:,1]),np.max(corners_cloud[:,1]), np.min(corners_cloud[:,2])-0.1,np.max(corners_cloud[:,2])+0.1]
leg_point_cloud_clone = leg_point_cloud.clone()
aruco_marker_cloud = leg_point_cloud.clone().crop(bounds= bound_box)
hits = []
for p in range (len(aruco_marker_cloud.points())):
    hits.append(leg_point_cloud.closestPoint(aruco_marker_cloud.points()[p],N=1, returnPointId=True))
leg_point_cloud_clone.deletePoints(hits).clean(tol=0.01)

stub_axis = gv.perpendicular_vector(leg_axis)
axis_p1 = leg_center+stub_axis*10 # define two points along the stub axis 
axis_p2 = leg_center-stub_axis*10

#define a configuration of the stub onto the leg
axes = np.array([gv.perpendicular_vector(stub_axis), leg_axis, stub_axis]) #refrence systemn for x,y,z axis for transformation of the stub
rotation = np.asarray([40,0,0]) #rotation in degrees
translation = np.array(leg_axis*0.5) #translation along the axes defined two lins above

T = np.identity(4)
T[:3,3] = translation

#create a leg_mesh
leg_mesh = v.pointcloud.delaunay2D(leg_c_d.points, mode='fit').subdivide(3) # create leg mesh from blender point cloud
leg_mesh = leg_mesh.triangulate().alpha(0.2)
intersect_c = leg_mesh.intersectWithLine(axis_p1, axis_p2)

#synthetic stub generation
stub  = v.shapes.Cylinder(pos=aruco_point, r=0.3, height=2, axis=stub_axis, c='teal3', alpha=1, cap=False, res=1000)
stub_mesh = stub.triangulate().alpha(0.2)
stub_config = stub_mesh.clone()
for i in range(len(axes)):
    stub_config.rotate(rotation[i],axis=axes[i],point=aruco_point)
stub_config.applyTransform(T)

intersect_curve = leg_mesh.intersectWith(stub_config).join(reset=True) # fix the boundary in the array and intersect two mesh
spline3d = v.shapes.Spline(intersect_curve).lw(3)

#compute curvature of the spline by fitting circle to a set of points around the intersection curve
spline = intersect_curve.clone()
#spline.projectOnPlane()
points = spline.points()
print("spline points", spline.N())

fitpts, circles, curvs = [], [], []
n = 250                   # nr. of points to use for the fit
points2 = np.append(points, points[0:n], axis=0) #append so the last points in the points2 are the first ones of points
for i in range(spline.NPoints()):
    pts = points2[i:i+n]
    center, R, normal = v.pointcloud.fitCircle(pts)
    circles.append(R)
    curvs.append(np.sqrt(1/R))

spline.lw(8).cmap('rainbow', curvs).addScalarBar3D(title='\pm1/\sqrtR')

isc = intersect_curve.clone().shift(0,0,0.01).pointSize(3).c('k') # to make it visible
pcurv = v.pyplot.plot(curvs, '-')  # plot curvature values

# smooth
smooth = gaussian_filter1d(curvs, 100)

# compute second derivative
smooth_d2 = np.gradient(np.gradient(smooth))

# find switching points
infls = np.where(np.diff(np.sign(smooth_d2)))[0]

p = v.Points(points[infls[:]]).c('r').ps(10)
t= []
for i in range(0, len(infls)):
    t.append([infls[i], curvs[infls[i]]])

sorted_curvs = np.sort(curvs)
max_curvature = points[np.where(curvs == sorted_curvs[-1])[0][0]]
min_curvature = points[np.where(curvs == sorted_curvs[0])[0][0]]
p_1 = v.Points([max_curvature], r=10).c('r').ps(20)
p_min = v.Points([min_curvature], r=10).c('g').ps(20)

#compute the greatest distance from the centerpoint of the stub to the curve
center2d = (aruco_point+translation)[0:2]
l = list(intersect_curve.points()[:,0:2])
l.sort(key=lambda coord: gv.dist(center2d[0], center2d[1] , coord[0], coord[1]))
sorted_arr = np.array(l)
direct = (-sorted_arr[-1]+center2d[0:2])
a1 = sorted_arr[-1] -direct
a2 = center2d[0:2]+3*direct
line2d  = v.shapes.Line(a1,a2)

a2_3d = np.append(a2, 5)
a1_3d = np.append(a1,5)
a1 = np.append(a1,0)
a2 = np.append(a2,0)

line3d = v.shapes.Line(a1_3d, a2_3d)
Ribbon = v.shapes.Ribbon(line2d, line3d)
Ribbon.triangulate()
r = v.shapes.Line(a1, a1_3d)
center_line = leg_mesh.intersectWith(Ribbon).join(reset=True)

hits = []
for i in center_line.points():
    cpt = spline3d.closestPoint(i)
    if v.mag2(i-cpt)<0.00001:
        hits.append(i)
# print(array(hits))
hits = v.Points(hits, r=10, c='r')

toe_t  = v.shapes.Text3D('X', pos=hits.points()[0] , s=0.08, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
heel_t  = v.shapes.Text3D('X', pos=hits.points()[1] , s=0.08, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)
center_t  = v.shapes.Text3D('X', pos=aruco_point+translation , s=0.08, font='', hspacing=0.1, vspacing=1.5, depth=-0, italic=False, justify='centered', c='8', alpha=10, literal=True)

# create an projected image
blank_image = np.zeros((1200,1920,3), np.uint8)
centerpoints = center_t.clone()
toe = toe_t.clone()
heel = heel_t.clone()
centerpoints.subdivide(5, method=1)
toe.subdivide(5, method=1)
heel.subdivide(5, method=1)
centerpoints_2d = gv.point_to_xy(centerpoints.clone().applyTransform(t_ext).points(),proj_intr)
toe_2d = gv.point_to_xy(toe.clone().applyTransform(t_ext).points(),proj_intr)
heel_2d = gv.point_to_xy(heel.clone().applyTransform(t_ext).points(),proj_intr)
splinepoints_2d = gv.point_to_xy(spline3d.clone().applyTransform(t_ext).points(),proj_intr)
centerline_points_2d = gv.point_to_xy(center_line.clone().applyTransform(t_ext).points(),proj_intr)



for i in range (0, len(splinepoints_2d)):
    blank_image[splinepoints_2d[i][1]][splinepoints_2d[i][0]]= [225,128,255]

#for i in range (0, len(centerline_points_2d)):
    #blank_image[centerline_points_2d[i][1]][centerline_points_2d[i][0]]= [225,128,255]

gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    cv2.drawContours(blank_image, [c], -1, (255,128,255), thickness=15)

for i in range (0, len(centerpoints_2d)):
    blank_image[centerpoints_2d[i][1]][centerpoints_2d[i][0]]= [220,20,60]

for i in range (0, len(toe_2d)):
    blank_image[toe_2d[i][1]][toe_2d[i][0]]= [0,0,255]

for i in range (0, len(toe_2d)):
    blank_image[heel_2d[i][1]][heel_2d[i][0]]= [0,255,0]

pic =v.Picture(blank_image)

print("#done")