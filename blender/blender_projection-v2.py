import numpy as np
import open3d as o3d
import cv2
import vedo as v
import geometry_vision as gv
from scipy.ndimage import gaussian_filter1d

no =3

#load intrinsics, extrinsics, pointcloud from the Blender setup
cam_int = np.genfromtxt("blender/scenes/csv/K_camera.csv", delimiter=",") # camera intrinsics
cam_res = [1920, 1200]
proj_intr = np.genfromtxt("blender/scenes/csv/K_projector.csv", delimiter=",") #projector intrinsics
proj_res = [1920, 1200]
t_ext = np.genfromtxt("blender/scenes/csv/T_1_2.csv", delimiter = ",") # Transformation from cam to proj

intr = o3d.open3d.camera.PinholeCameraIntrinsic(cam_res[1], cam_res[0], fx=cam_int[0][0], fy=cam_int[1][1], cx=cam_int[0][2], cy=cam_int[1][2]) #intrinsic parameters in open3d
leg_c, img_before, z = gv.pointcloud_blender(f'blender/scenes/result/{no}/pointcloud.exr', cam_res[0], cam_res[1],intr, mode="simple") #read rendered pointcloud from Blender
leg_white, img_white, z_white = gv.pointcloud_blender(f'blender/scenes/result/{no}/white.exr', cam_res[0], cam_res[1],intr, mode="simple") #read rendered pointcloud from Blender
leg_c_d = leg_c.voxel_down_sample(voxel_size=0.2) # downsample pointcloud
leg_c_d_m = leg_c.voxel_down_sample(voxel_size=0.06) # downsample pointcloud
leg_axis, leg_center, leg_r, fit_error = gv.fit(np.asarray(leg_c_d.points)) #cylinder fitting to the captured pointcloud

o3d.io.write_point_cloud("blender/leg_c.ply", leg_c)
leg_point_cloud = v.load("blender/leg_c.ply")

##add aruco mark to have a reference point
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco, corners = gv.read_aruco_mark(img_before, dictionary, show=False)
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
rotation = np.asarray([0,-30,0]) #rotation in degrees
translation = np.array(leg_axis*0.7) #translation along the axes defined two lins above

T = np.identity(4)
T[:3,3] = translation

#create a leg_mesh
leg_mesh = v.pointcloud.delaunay2D(leg_c_d_m.points, mode='fit').subdivide(3) # create leg mesh from blender point cloud
leg_mesh = leg_mesh.triangulate().alpha(0.2)
intersect_c = leg_mesh.intersectWithLine(axis_p1, axis_p2)

#synthetic stub generation
stub  = v.shapes.Cylinder(pos=aruco_point, r=0.35, height=2, axis=stub_axis, c='teal3', alpha=1, cap=False, res=1000)
stub_concentric  = v.shapes.Cylinder(pos=aruco_point, r=0.35+0.05, height=2, axis=stub_axis, c='teal3', alpha=1, cap=False, res=800)
stub_mesh = stub.triangulate().alpha(0.2)
stub_concentric_mesh = stub_concentric.triangulate().alpha(0.2)
stub_config = stub_mesh.clone()
stub_concentric_config = stub_concentric_mesh.clone()
for i in range(len(axes)):
    stub_config.rotate(rotation[i],axis=axes[i],point=aruco_point)
    stub_concentric_config.rotate(rotation[i],axis=axes[i],point=aruco_point)
stub_config.applyTransform(T)
stub_concentric_config.applyTransform(T)

intersect_curve = leg_mesh.intersectWith(stub_config).join(reset=True) # fix the boundary in the array and intersect two mesh
intersect_concentric_curve = leg_mesh.intersectWith(stub_concentric_config).join(reset=True)

spline3d = v.shapes.Spline(intersect_curve).lw(3)
spline3d_concentric = v.shapes.Spline(intersect_concentric_curve)

grind_surf = leg_mesh.clone()
grind_surf.cutWithMesh(stub_concentric_config, invert = False) #Alternative use cutWithCylinder which much faster
grind_surf.cutWithMesh(stub_config, invert = True)

splinepoints_2d = gv.point_to_xy(spline3d.clone().applyTransform(t_ext).points(),proj_intr)
splinepoints_concentric_2d = gv.point_to_xy(spline3d_concentric.clone().applyTransform(t_ext).points(),proj_intr)

blank_image = np.full((1200, 1920,3), 255)     
img =blank_image.copy()
img2 = cv2.fillPoly(img, [splinepoints_concentric_2d], [0,255,0])
blank_image = cv2.fillPoly(img2, [splinepoints_2d], [255,255,255])
for i in range (0, len(splinepoints_2d)):
    blank_image[splinepoints_2d[i][1]][splinepoints_2d[i][0]]= [0,255,0]
blank_image = v.Picture(blank_image)
blank_image.write(f'blender/scenes/result/{no}/img0.png')
black_image = np.zeros((1200,1920,3), np.uint8) 

leg_p1 = leg_center +0.300*leg_axis
leg_p2 = leg_center -0.300*leg_axis
axis = v.shapes.Line(leg_p1,leg_p2)
axis_2 = v.shapes.Line(axis_p1,axis_p2)
ref_y = v.shapes.Arrow(aruco_point,np.array([aruco_point]+axes[0]*0.300)[0]).c('g')
ref_x = v.shapes.Arrow(aruco_point,np.array([aruco_point]+axes[1]*0.300)[0]).c('r')
ref_z = v.shapes.Arrow(aruco_point,np.array([aruco_point]+axes[2]*-0.300)[0]).c('b')
ref = v.Assembly(ref_x, ref_y, ref_z)
test_cyl =v.shapes.Cylinder(pos=leg_center, r=leg_r, height=1, axis=leg_axis, c='teal3', alpha=1, cap=False, res=800)
#create a self made coordinate system
x= v.shapes.Arrow([0,0,0],[0.2,0,0]).c('r')
y= v.shapes.Arrow([0,0,0],[0,0.2,0]).c('g')
z= v.shapes.Arrow([0,0,0],[0,0,0.2]).c('b')
cam_sys = v.Assembly(x,y,z)
x_p = x.clone().applyTransform(t_ext)
y_p = y.clone().applyTransform(t_ext)
z_p = z.clone().applyTransform(t_ext)
proj_syst = v.Assembly(x_p,y_p,z_p)


leg_c_after, img_after, xyz_projection = gv.pointcloud_blender(f'blender/scenes/result/{no}/pointcloud_after.exr', cam_res[0], cam_res[1],intr, mode="complete") #read rendered pointcloud from Blender
section_cut_points, thresh = gv.morphing(img_after, img_white,xyz_projection,  np.ones((8,8),np.uint8), show= False)

section_cut_points, thresh = gv.morphing(img_after, img_white,xyz_projection,  np.ones((8,8),np.uint8), show= False)
projected_section = v.Points(section_cut_points).clean(tol=0.002)
projected_section_p = v.Points(section_cut_points).clean(tol=0.002)
projected_section_p.alignTo(grind_surf, rigid = True)
grind_surf_projected  = v.recoSurface(projected_section, radius= 0.009, dims=(150,150,500))
align_transformation = np.eye(4)
projected_section_p.GetMatrix().DeepCopy(align_transformation.ravel(), projected_section_p.GetMatrix())
grind_surf_projected_signed = grind_surf_projected.clone()
grind_surf_projected_signed.distanceToMesh(grind_surf, signed=True, negate=False)
grind_surf_projected_signed.addScalarBar(title='Signed\nDistance')
signed_dist = grind_surf_projected_signed.getPointArray("Distance")
signed_dist_avg = np.average(signed_dist)


plot = []
config_result_2 = []
config_result_2.append([leg_mesh,stub_config, stub_concentric_config, intersect_curve, intersect_concentric_curve, spline3d, spline3d_concentric, aruco_marker_cloud, v.Points(intersect_c), v.Points([intersect_c[0]+translation])])
plot.append([grind_surf, grind_surf_projected, projected_section, projected_section_p, grind_surf_projected_signed, align_transformation, signed_dist, signed_dist_avg, thresh,leg_c_after, blank_image])



'''
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
'''
print("done")