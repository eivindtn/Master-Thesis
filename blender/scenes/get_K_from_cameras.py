import bpy
from math import *
import numpy as np
from mathutils import *
import os
import sys



def get_calibration_matrix_K_from_blender(cam, mode='simple'):

    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = bpy.data.cameras[cam]

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':
        #bpy.data.cameras["Camera.002"].lens
        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0 # only use rectangular pixels

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K

def writeCSV(filename, matrix):
    np.savetxt(filename, matrix, delimiter=",", fmt='% s')

def render(filename, output_dir, res_x, res_y):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = "GPU"
    # Set the device_type
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
    bpy.context.scene.render.image_settings.file_format='OPEN_EXR'
    bpy.context.scene.render.filepath = os.path.join(output_dir, filename + ".exr")
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.ops.render.render(write_still=True)

#Map projector light to have equal field of view as a camera
camdata = bpy.data.cameras['Camera-2']
focal_length = camdata.lens
sensorwidth_x = camdata.sensor_width
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1200
res_x =  bpy.context.scene.render.resolution_x
res_y= bpy.context.scene.render.resolution_y

x_scale = focal_length/sensorwidth_x
y_scale = (focal_length/sensorwidth_x)*(res_x/res_y)

node_tree = bpy.data.lights['Projector'].node_tree
node_tree.nodes['Mapping'].inputs[3].default_value = [x_scale,y_scale, 1]

T_1 = (bpy.data.objects["Camera-1"].matrix_world)
T_2 = (bpy.data.objects["Camera-2"].matrix_world)
#print(get_t('Camera-1'))

T_1 = np.asarray(np.round(T_1,4))

T_2 = np.asarray(np.round(T_2,4))

T_1_2 = np.linalg.inv(T_1) @ T_2

#get the intrinsics parameters from the blender cameras
K_camera  = get_calibration_matrix_K_from_blender('Camera-1',mode='complete')
K_projector = get_calibration_matrix_K_from_blender('Camera-2',mode='complete')
#T_camera = get_3x4_RT_matrix_from_blender(bpy.data.cameras['Camera-2'])

#write to location
writeCSV("C:\\Users\\eivin\\Desktop\\NTNU master-PUMA-2019-2021\\4.studhalvår\\Repo\Master-Thesis\\blender\\scenes\\csv\K_projector.csv", K_projector)
writeCSV("C:\\Users\\eivin\\Desktop\\NTNU master-PUMA-2019-2021\\4.studhalvår\\Repo\Master-Thesis\\blender\\scenes\\csv\K_camera.csv", K_camera)
writeCSV("C:\\Users\\eivin\\Desktop\\NTNU master-PUMA-2019-2021\\4.studhalvår\\Repo\Master-Thesis\\blender\\scenes\\csv\T_1_2.csv", T_1_2)

#render a point cloud via openexr
scene = bpy.context.scene
scene.camera = bpy.data.objects["Camera-1"]
render("Camera-1", "exr", res_x, res_y)
bpy.context.scene.render.filepath = os.path.join("C:\\Users\\eivin\\Desktop\\NTNU master-PUMA-2019-2021\\4.studhalvår\\Repo\Master-Thesis\\blender\\scenes\\exr", ('pointcloud.exr'))
bpy.ops.render.render(write_still = True)
