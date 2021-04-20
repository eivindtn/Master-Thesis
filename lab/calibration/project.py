import numpy as np
import cv2
import screeninfo
import zivid


def init_proj(window_name, screen_id):
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    cv2.moveWindow(window_name, screen.x -1, screen.y-1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
    return width, height
    

def main():

    # init projector
    PROJ_WIN = "projector_win"
    SCREEN_ID = 1
    cv2.namedWindow(PROJ_WIN, cv2.WND_PROP_FULLSCREEN)
    proj_w, proj_h = init_proj(PROJ_WIN, SCREEN_ID)
    print(proj_w, proj_h)


    #test image
    proj_im = np.zeros((proj_h,proj_w,3), np.uint8)
    img = cv2.imread('chessboard_7x13.png')


    cap = cv2.VideoCapture(2)

    while(True):
        # Capture frame-by-frame
        #ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #proj_im = cv2.resize(gray, (proj_w, proj_h))

        cv2.imshow(PROJ_WIN, img)
        

        # Display the resulting frame
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("done")
    
if __name__ == '__main__':
    main()