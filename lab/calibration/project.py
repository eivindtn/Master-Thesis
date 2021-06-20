import numpy as np
import glob
import cv2
import time
import screeninfo
import datetime 
import zivid



def init_proj(window_name, screen_id):
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    cv2.moveWindow(window_name, screen.x -1, screen.y-1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
    return width, height
def capture_2d(filepath):
    app = zivid.Application()
    camera = app.connect_camera()
    settings_2d = zivid.Settings2D()
    settings_2d.acquisitions.append(zivid.Settings2D.Acquisition())
    settings_2d.acquisitions[0].exposure_time = datetime.timedelta(microseconds=30000)
    settings_2d.acquisitions[0].aperture = 5.0
    settings_2d.acquisitions[0].brightness = 0
    settings_2d.acquisitions[0].gain = 1.0
    settings_2d.processing.color.balance.red = 1.0
    settings_2d.processing.color.balance.green =1.0
    settings_2d.processing.color.balance.blue = 1.0
    settings_2d.processing.color.gamma = 1.0
    settings_2d.acquisitions
    with camera.capture(settings_2d) as frame_2d:
        image = frame_2d.image_rgba()
        rgba = image.copy_data()
        image_file = filepath
        image.save(image_file)
def normalize_image(dark_img, ligt_img, img):
    normalized = np.ones_like(img)
    dark_img= np.asarray(dark_img)
    ligt_img = np.asarray(ligt_img)
    img = np.asarray(img)
    min_img = 2*np.ones_like(img)
    max_img = 255*np.ones_like(img)
    for row in range(len(img[:,0])):
        for cols in range(len(img[0,:])):
            if abs(ligt_img[row, cols]-dark_img[row, cols]) > 10**-2:
                normalized[row, cols] = int(((img[row, cols]-dark_img[row, cols])*((max_img[row, cols]-min_img[row, cols])/(ligt_img[row, cols]-dark_img[row, cols])))+min_img[row, cols])
            else:
                normalized[row, cols] = 0
    return normalized


def main():
    graycode_pattern = glob.glob("lab\calibration-mesh-brown\sample_data\graycode_pattern\*.png")
    # init projector
    c = 3 #capture iteratio
    for j in range (0,5):
        for i in range(len(graycode_pattern)):
            PROJ_WIN = "projector_win"
            SCREEN_ID = 1
            cv2.namedWindow(PROJ_WIN, cv2.WND_PROP_FULLSCREEN)
            proj_w, proj_h = init_proj(PROJ_WIN, SCREEN_ID)
            img = cv2.imread(graycode_pattern[i])

            cv2.imshow(PROJ_WIN, img)
            #time.sleep(2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            capture_2d(f'lab\calibration-mesh-brown\sample_data\capture_{j}\graycode_{graycode_pattern[i][-6:-4]}.png')
        
            #time.sleep(1)
        change = cv2.imread('lab\calibration-mesh-brown\sample_data\change.jpg')
        cv2.imshow(PROJ_WIN, change)
        cv2.waitKey(1)
        time.sleep(10)
    cv2.destroyAllWindows()

    print("done")
    # cd C:\Users\eivin\Desktop\NTNU master-PUMA-2019-2021\4.studhalvår\Repo\Master-Thesis\lab\calibration-mesh-brown\sample_data
    # python ../calibrate.py 1080 1920 8 6 35 1 -black_thr 40 -white_thr 5 - camera_config.json
if __name__ == '__main__':
    main()