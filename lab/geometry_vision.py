import Imath
import array     #Package for importing an .exr format into python 3
import OpenEXR

import numpy as np
import open3d as o3d
import math
import matplotlib.pyplot as plt
import cv2
import zivid
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity
from scipy.spatial.transform import Rotation as Rotxyz



from cylinder_fitting import fit
"""
Written by: https://github.com/xingjiepan/cylinder_fitting/tree/master/cylinder_fitting
Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf.
    Example: testfit = fit(np.array(cyl_points))
Parameters:
    Cylinder points
Return:
    fit[0]: cylinder axis
    fit[1]: cylinder center
    fit[2]: cylinder radius
    fit[3]: fit_error 
"""

def pixel_to_point(depth, point, intrinsics, depth_scale=1000):

    """ Get 3D world coordinates of 2D image cordinates in camera
    Parameters:
        point: pixel cordinate u,v
        depth scale: depends on the output unit
        intrinsics - np.array depth camera intrinsics matrix:
                [[fx, 0, ppx],
                 [0, fy, ppy],
                 [0, 0, 1]]
    Returns:
        x, y, z: coordinates in world
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    ppx = intrinsics[0, 2]
    ppy = intrinsics[1, 2]
    u, v = point
    x = (u - ppx) / fx
    y = (v - ppy) / fy
    z = depth / depth_scale
    x = x * z
    y = y * z
    return np.array([x, y, z])

def point_to_xy(points, intrinsics):
    """ Get 2D image coordinates of 3D point in camera

    Parameters:
        point: x,y,z vector
        np.array - depth camera intrinsics matrix:
                [[fx, 0, ppx],
                 [0, fy, ppy],
                 [0, 0, 1]]

    Returns:
        x, y: coordinates in image, might want to round
    """

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    ppx = intrinsics[0, 2]
    ppy = intrinsics[1, 2]
    
    xy_points=[]
    for p in points:
        m = p[0] / p[2]
        n = p[1] / p[2]

        x = m * fx + ppx
        y = n * fy + ppy
        
        xy = [int(x), int(y)]
        xy_points.append(xy)
    
    #xy_points = np.unique(xy_points, axis=0) <  1
    
    return np.array(xy_points)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    taken from: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_matrix(axis, theta):
    """
    taken from:https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def create_transformation(axes,rotation, rotpoint, position, translation= None):
    """ Create a transformation matrix from rotating an object around an axis through a rotation point
    Parameters:
    axes: Three axes which defines a refrence system of the object
    rotation: Apply a rotion angle around each of the axes
    rotpoint: Rotation point
    position: The position in the global coordinate system
    Translation: Add this later
    
    Returns:
        T: the transformation matrix for all the three rotations
    """
    T = np.eye(4)
    rot = np.identity(3)
    for i in range(len(axes)):
        R = rotation_matrix(axes[i],np.radians(rotation[i]))
        rot= R@rot
        T[:3,:3] = rot
        T[:3,3] = [0,0,0]#np.dot(R, np.array(position-rotpoint)) +rotpoint + translation
    return T

def perpendicular_vector(v):
    """ Find the perpendicular vector for a vector
    Parameters: v: 3d vector
    return: perpundcular vector to v
    """
    if np.round(v[1]) == 0 and np.round(v[2]) == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

def transformation_r_t(R,T):
    """ Create a transformation matrix from R ant t
    Parameters: r, rotation matrix 
                t, 3d translation vector
    return: transformation matrix
    """
    return np.asarray([[R[0][0],R[0][1],R[0][2], T[0]],
                        [R[1][0],R[1][1],R[1][2], T[1]],
                        [R[2][0],R[2][1],R[2][2],T[2]],
                        [0,0,0, 1]])

def pointcloud_blender(inputEXR, resx,resy, intrinsics, mode= None):
    """ Get a pointclooud from a rendered camera image from Blender
    Parameters:
    inputEXR: Render an image Blender and save it as a .exr file
    resx: Blender camera resoultion along x
    resy: Blender camera resolution along y
    intrinsics: intrinsics - np.array depth camera intrinsics matrix:
                [[fx, 0, ppx],
                 [0, fy, ppy],
                 [0, 0, 1]]
    mode: set mode = 'complete' to get a pixel pointcloud 1:1 correnspondent

    Returns:
        pcd, an open3d pointcloud containing points and rgb image   

    """
    exr_img = OpenEXR.InputFile(inputEXR) #ReadExr file
    cs = list(exr_img.header()['channels'].keys())  # channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    img_data = [np.array(array.array('f', exr_img.channel(c, FLOAT))) for c in cs]
    img_data = [d.reshape(resy,resx) for d in img_data]
    rgb = np.concatenate([img_data[i][:, :, np.newaxis] for i in [3, 2, 1]], axis=-1)
    rgb /= np.max(rgb)  # this will result in a much darker image
    np.clip(rgb, 0, 1.0)  # to better visualize as HDR is not supported?

    img = o3d.geometry.Image((rgb * 255).astype(np.uint8)) #create image
    depth = o3d.geometry.Image((img_data[-1] * 1000).astype(np.uint16)) #set depth data to mm
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False) # the scaling need to be set to 1000 to have mm
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics) # return a pointcloud in open3d format
    z = np.asarray(depth)
    if mode == "complete":
        pixels_xyz_corr = np.empty((resy,resx,3))
        for i in range(1080):
           for j in range(1920):
            pixels_xyz_corr[i][j] = pixel_to_point(z[i][j],[i,j],intrinsics.intrinsic_matrix, depth_scale=10000) # set the depth scal to the same as above
        return pcd, np.asarray(img),np.asarray(pixels_xyz_corr)
    return pcd,np.asarray(img),z

def make_cylinder(radius, length, nlength, alpha, nalpha, center, orientation):
    """ Create a cylinder along an axis
    Parameters:
    radius: radius of cylinder
    length: length of cylinder
    nlength: number of points around the circle
    alpha: how much angle on the cylinder
    nalpha: how many layer of points
    center: center of the cylinder
    orientation: axis direction of the cylinder

    Return: cylinder points
    """
    #Create the length array
    I = np.linspace(0, length, nlength)

    #Create alpha array avoid duplication of endpoints
    #Conditional should be changed to meet your requirements
    if int(alpha) == 360:
        A = np.linspace(0, alpha, num=nalpha, endpoint=False)/180*np.pi
    else:
        A = np.linspace(0, alpha, num=nalpha)/180*np.pi

    #Calculate X and Y
    X = radius * np.cos(A)
    Y = radius * np.sin(A)

    #Tile/repeat indices so all unique pairs are present
    pz = np.tile(I, nalpha)
    px = np.repeat(X, nlength)
    py = np.repeat(Y, nlength)

    points = np.vstack(( pz, px, py )).T

    #Shift to center
    #shift = np.array(center) - np.mean(points, axis=0)
    #points += shift

    #Orient tube to new vector

    #Grabbed from an old unutbu answer
    def rotation_matrix(axis,theta):
        a = np.cos(theta/2)
        b,c,d = -axis*np.sin(theta/2)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    ovec = orientation / np.linalg.norm(orientation)
    cylvec = np.array([1,0,0])

    #if np.allclose(cylvec, ovec):
        #return points

    #Get orthogonal axis and rotation
    oaxis = np.cross(ovec, cylvec)
    rot = np.arccos(np.dot(ovec, cylvec))

    R = rotation_matrix(oaxis, rot)
    return (points.dot(R))


def clockwiseangle_and_distance(point, origin, refvec):
    """ order a set of points in a clockwise direction
    taken from: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
    """
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def dist(i,j,ip,jp): 
    return np.sqrt((i-ip)**2+(j-jp)**2)

def order_boundary(msh):
    poly = msh.join().polydata(False)
    poly.GetPoints().SetData(poly.GetCell(0).GetPoints().GetData())
    return msh

def interpolate_xyz(interpolate_number, array, u, v):
        xyz = array[v-interpolate_number:v+interpolate_number+1, u-interpolate_number:u+interpolate_number+1]
        x_indx,y_indx,z_indx = np.meshgrid(np.arange(0, np.shape(xyz)[0]),np.arange(0,np.shape(xyz)[1]), np.arange(0,np.shape(xyz)[2]))
            
        array_masked = np.ma.masked_invalid(xyz)
        valid_xs = x_indx[~array_masked.mask]
        valid_ys = y_indx[~array_masked.mask]
        valid_zs = z_indx[~array_masked.mask]
        validarr = array_masked[~array_masked.mask]
            
        xyz_interp = griddata((valid_xs, valid_ys, valid_zs), validarr.ravel(),
                                        (x_indx, y_indx, z_indx), method='nearest')
        return xyz_interp[interpolate_number][interpolate_number]

def create_aruco_mark(dictionary, name):
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, 22, 200, markerImage, 1)
    cv2.imwrite(name, markerImage)

def read_aruco_mark(img, dictonary, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictonary, parameters= parameters)
    rgb_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

    plt.figure()
    plt.imshow(rgb_markers)
    for i in range(len(ids)):
        c = corners[i][0]
        # for 2D visualization, plot middle of the marker
        x = c[:,0].mean()
        y = c[:,1].mean()
        plt.plot([x],[y], "o", label = "id={0}".format(ids[i]))
    plt.legend()
    if show:
        plt.show()
    return [x,y], corners

def find_aruco_point(aruco_marker, xyz, interpolate_number,rgba):
    corners_cloud = []
    dictionary = cv2.aruco.Dictionary_get(aruco_marker)
    aruco, corners = read_aruco_mark(rgba[:, :, 0:3], dictionary, show= True)
    aruco_point = xyz[int(aruco[1]),int(aruco[0])]
    if np.isnan(aruco_point).any() == True:
        print(f'Could not find the aruco marker in the point cloud, interpolate to find with the {interpolate_number} nearest pixels.')
        aruco_point  = interpolate_xyz(interpolate_number, xyz,int(aruco[0]),int(aruco[1]))
    for i in range (len(corners[0][0])):
        corner_i = xyz[int(corners[0][0][i][1])][int(corners[0][0][i][0])]
        if np.isnan(corner_i).any() == True:
            corners_cloud.append(interpolate_xyz(interpolate_number, xyz, int(corners[0][0][i][0]), int(corners[0][0][i][1])))
        else:
             corners_cloud.append(corner_i)
    corners_cloud = np.asarray(corners_cloud)
    return aruco_point, corners_cloud
def delete_aruco_marker_from_point_cloud(leg_point_cloud, corners_cloud, bound_x, bound_y, bound_z):
    bound_box = [np.min(corners_cloud[:,0])-bound_x[0],np.max(corners_cloud[:,0])+bound_x[1],np.min(corners_cloud[:,1])-bound_y[0],np.max(corners_cloud[:,1])+bound_y[1], np.min(corners_cloud[:,2])-bound_z[0],np.max(corners_cloud[:,2])+bound_z[1]]
    leg_point_cloud_clone = leg_point_cloud.clone()
    aruco_marker_cloud = leg_point_cloud.clone().crop(bounds= bound_box)
    hits = []
    for p in range (len(aruco_marker_cloud.points())):
        hits.append(leg_point_cloud.closestPoint(aruco_marker_cloud.points()[p],N=1, returnPointId=True))
    leg_point_cloud_clone.deletePoints(hits)
    return leg_point_cloud_clone, aruco_marker_cloud

def display_pointcloud(xyz, rgb):
    """Display point cloud.
    Args:
        rgb: RGB image
        xyz: X, Y and Z images (point cloud co-ordinates)
    Returns None"""
    xyz = np.nan_to_num(xyz).reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb / 255)

    visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)

    visualizer.get_render_option().background_color = (0, 0, 0)
    visualizer.get_render_option().point_size = 1
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.get_view_control().set_front([0, 0, -1])
    visualizer.get_view_control().set_up([0, -1, 0])

    visualizer.run()
    visualizer.destroy_window()

def load_zdf_frames(zdf, ply, visualization = False, downsample = False):
    app = zivid.Application()
    frame = zivid.Frame(zdf)
    frame.save(ply)
    point_cloud = frame.point_cloud()
    xyz = frame.point_cloud().copy_data("xyz")
    rgba = point_cloud.copy_data("rgba")
    if visualization == True:
        display_pointcloud(xyz, rgba[:, :, 0:3])
    if downsample == True:
        point_cloud.downsample(zivid.PointCloud.Downsampling.by2x2)
        xyz_donwsampled = point_cloud.copy_data("xyz")
        rgba_downsampled = point_cloud.copy_data("rgba")
        display_pointcloud(xyz_donwsampled, rgba_downsampled[:, :, 0:3])
        return xyz, rgba, xyz_donwsampled, rgba_downsampled

    return xyz, rgba

def display_rgb(rgb, title):
    """Display RGB image.
    Args:
        rgb: RGB image (HxWx3 darray)
        title: Image title
    Returns None
    """
    plt.figure()
    plt.imshow(rgb)
    plt.title(title)
    plt.show(block=False)


def display_depthmap(xyz):
    """Create and display depthmap.
    Args:
        xyz: X, Y and Z images (point cloud co-ordinates)
    Returns None
    """
    plt.figure()
    plt.imshow(
        xyz[:, :, 2],
        vmin=np.nanmin(xyz[:, :, 2]),
        vmax=np.nanmax(xyz[:, :, 2]),
        cmap="viridis",
    )
    plt.colorbar()
    plt.title("Depth map")
    plt.show(block=False)

def find_image_structural_simularity(img1, img2, xyz, show= True):
    # convert the images to grayscale
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if show == True:
        plt.imshow(thresh)
        plt.show()
    y,x = np.nonzero(thresh)
    points = []
    for i in range (len(x)):
        points.append(xyz[y[i],x[i]])
    return points,thresh

def morphing(img1, img2,xyz, kernel, show= True):
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image = grayA-grayB
    morph = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    y,x = np.nonzero(morph)
    points = []

    if show == True:
        plt.imshow(morph)
        plt.show()

    for i in range (len(x)):
        if np.invert(np.isnan(xyz[y[i],x[i]])).any() == True:
            points.append(xyz[y[i],x[i]])
    return points, morph 
def R_to_eul(R):
    r = Rotxyz.from_matrix(R)
    return r