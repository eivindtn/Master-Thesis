import numpy as np
import cv2
import zivid
import open3d as o3d
import vedo as v
import matplotlib.pyplot as plt

from pathlib import Path

from sample_utils.paths import get_sample_data_path
from sample_utils.settings_from_file import get_settings_from_yaml
import screeninfo

def capture_w_yaml(settings_file, output_path, ply_path):
    app = zivid.Application()
    print("Connecting to camera")
    camera = app.connect_camera()

    #settings_file = Path() / get_sample_data_path() 
    print(f"Configuring settings from file: {Path('C:/Users/eivin/Desktop/NTNU master-PUMA-2019-2021/4.studhalvår/Repo/Master-Thesis/lab/settings/lab_settings.yml')}")
    settings = get_settings_from_yaml(Path(settings_file))

    print("Capturing frame")
    with camera.capture(settings) as frame:
        print(f"Saving frame to file: {output_path}")
        frame.save(output_path)
        frame.save(ply_path)
        point_cloud = frame.point_cloud()
        xyz = frame.point_cloud().copy_data("xyz")
        rgba = point_cloud.copy_data("rgba")
    return v.load(ply_path), xyz, rgba

def load_zdf_frames(zdf, ply, visualization = False, downsample = False):
    app = zivid.Application()
    frame = zivid.Frame(zdf)
    #frame.save(ply)
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
    return v.load(ply),xyz, rgba

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


def init_proj(window_name, screen_id):
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    cv2.moveWindow(window_name, screen.x -1, screen.y-1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    return width, height