import numpy as np
import cv2
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import imutils

# load the two input images
imageA = cv2.imread('lab/results/Experiment_2/leg.png')
imageB = cv2.imread('lab/results/Experiment_2/config_0/1.png')

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB,gaussian_weights = True, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

test_image = np.zeros((1200,1944,3), np.uint8)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in cnts[0]]
sort_area = sorted(areas)
index = np.where(np.asarray(areas) ==sort_area[-5]) 
for c in cnts[0][index[0][0]]:
    cv2.drawContours(test_image, [c], -1, (255,128,255), thickness=2)
plt.imshow(test_image)
plt.show()
print('done')