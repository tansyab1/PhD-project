import os 
import cv2
import numpy as np
import csv
from tqdm import tqdm

# calculate the Spatial Information and Temporal Information of a video

def calcualteSITI(videoin):
    spatial = []
    temporal = []
    
    # read the video
    video = cv2.VideoCapture(videoin)
    # check if the video is opened
    if not video.isOpened():
        print("Error opening video file")
    
    # read the first frame
    # gray_old = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1):
        if i == 0:
            gray_old = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
        # read the frame
        ret, frame = video.read()
        # convert the frame to gray
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # normalize the gray image
            gray = gray/255
            # calculate the spatial information
            sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
            sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
            # discard the first and the last row and column 
            sobelx = sobelx[1:-1,1:-1]
            sobely = sobely[1:-1,1:-1]
            
            spatial.append(np.std(np.sqrt(sobelx**2+sobely**2)))
            
            # calculate the temporal information between two frames
            temporal.append(np.std(np.abs(gray[1:-1,1:-1]-gray_old[1:-1,1:-1])))
            gray_old = gray
            # print(spatial[-1], temporal[-1])
        
    return np.max(spatial), np.max(temporal)

# calculate the colorfulness and global contrast factor of a video
def calculateCGCF(videoin):
    colorfulness = []
    gcf = []
    
    # read the video
    video = cv2.VideoCapture(videoin)
    # check if the video is opened
    if not video.isOpened():
        print("Error opening video file")
    
    # read the first frame
    # gray_old = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1):
        if i == 0:
            frame_old = video.read()[1]
        # read the frame
        ret, frame = video.read()
        
        if frame is not None:
            # calculate the colorfulness
            colorfulness.append(colorfulness_measure(frame))
            # calculate the global contrast factor
            gcf.append(compute_global_contrast_factor(frame))
            # print(colorfulness[-1], gcf[-1])
        
    return np.max(gcf), np.max(colorfulness)

def colorfulness_measure(img):
    
    r,g,b = cv2.split(img)
    # convert to 0-1 range
    # r = r/255
    # g = g/255
    # b = b/255
    #  rg = R - G
    rg = np.abs(r - g)
    #  yb = 1/2(R + G) - B
    yb = np.abs(0.5 * (r + g) - b)
    
    stdRG = np.std(rg);
    meanRG = np.mean(rg);
    
    stdYB = np.std(yb);
    meanYB = np.mean(yb);
    
    stdRGYB = np.sqrt((stdRG)**2 + (stdYB)**2);
    meanRGYB = np.sqrt((meanRG)**2 + (meanYB)**2);
    
    C = stdRGYB + 0.3 * meanRGYB;
    return C
    
def compute_image_average_contrast(k, gamma=2.2):
    L = 100 * np.sqrt((k / 255) ** gamma )
    # pad image with border replicating edge values
    L_pad = np.pad(L,1,mode='edge')

    # compute differences in all directions
    left_diff = L - L_pad[1:-1,:-2]
    right_diff = L - L_pad[1:-1,2:]
    up_diff = L - L_pad[:-2,1:-1]
    down_diff = L - L_pad[2:,1:-1]

    # create matrix with number of valid values 2 in corners, 3 along edges and 4 in the center
    num_valid_vals = 3 * np.ones_like(L)
    num_valid_vals[[0,0,-1,-1],[0,-1,0,-1]] = 2
    num_valid_vals[1:-1,1:-1] = 4

    pixel_avgs = (np.abs(left_diff) + np.abs(right_diff) + np.abs(up_diff) + np.abs(down_diff)) / num_valid_vals

    return np.mean(pixel_avgs)

def compute_global_contrast_factor(img):
    if img.ndim != 2:
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gr = img

    superpixel_sizes = [1, 2, 4, 8, 16, 25, 50, 100, 200]

    gcf = 0

    for i,size in enumerate(superpixel_sizes,1):
        wi =(-0.406385 * i / 9 + 0.334573) * i/9 + 0.0877526
        im_scale = cv2.resize(gr, (0,0), fx=1/size, fy=1/size,
                              interpolation=cv2.INTER_LINEAR)
        avg_contrast_scale = compute_image_average_contrast(im_scale)
        gcf += wi * avg_contrast_scale

    return gcf
    

if __name__ == '__main__':
    # get the file list
    forder = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/fps5/cut/ref_videos/ref_videos"
    # file = "/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/test.mp4"
    
    # read all the files in the folder and save the result to a csv file
    
    for file in tqdm(os.listdir(forder)):
        if file.endswith(".mp4"):
            # print(os.path.join(forder, file))
            fullpath = os.path.join(forder, file)
            si, ti = calcualteSITI(fullpath)
            gcf, c = calculateCGCF(fullpath)
            
            # save the result to a csv file
            with open('result.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([file, si, ti, gcf, c])
                f.close()