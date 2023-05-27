import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file', default='/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Data/labelled_video/slow_traffic_small.mp4')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (175, 175),
                  maxLevel = 5,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create color white for drawing
color = (255,255,255)

def find_centroid(img):
    # find the contours of the image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # find the centroid of the contours
    M = cv.moments(img)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# read the second frame
ret, old_frame2 = cap.read()
old_gray2 = cv.cvtColor(old_frame2, cv.COLOR_BGR2GRAY)

diff_images = cv.absdiff(old_gray, old_gray2)
_,binary_images = cv.threshold(diff_images, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
centroid = find_centroid(binary_images)

# save the binary image
cv.imwrite('binary_images.png', binary_images)
# save the diff image
cv.imwrite('diff_images.png', diff_images)

# print(centroid)

# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# print(p0)
p0 = np.array([[[centroid[0], centroid[1]]]], dtype=np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
# save the video
out = cv.VideoWriter('opticalflow.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        # print('p1 is not None')
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # print('i: ', i)
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color, -1)
    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    out.write(img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)


    

cv.destroyAllWindows()