import cv2
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import image_slicer
from PIL import Image
from scipy import signal as sg
import glob
import os
import shutil

def afficher_image(img) :

    cv2.imshow('ImageJolie',img)
   # key = cv2.waitKey(0)
    #cv2.destroyAllWindows()

vidcap = cv2.VideoCapture('videos/colon_normal_puis_MAyo_2.avi')
vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,2)
success,image = vidcap.read()
length = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
a,b,c = image.shape

mat = np.zeros((a,b,c, 201))

#Constuire la masque
frame=1400
for i in range(-100, 101):
    vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame + i)
    success, image = vidcap.read()
    mat[:, :, :, i + 100] = image

m = np.nanmean(mat,3)
m = np.uint8(m)

#  fig = plt.figure()
#  ax1 = fig.add_subplot(2,1,1)
#plt.imshow(m[:,:,2],cmap='gray')
#plt.title('image moyenne')

#Algorithme top-hat :calculer le residu d ouverture
kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
m_topHat = cv2.morphologyEx(m,cv2.MORPH_TOPHAT,kernel)
#ax2 = fig.add_subplot(2,1,2)
plt.imshow(m_topHat)
#plt.show()
cat1 = "Figures/ChercherTexte"
cat2 = "Figures/ChercherTexte/FrameMasque"
if not os.path.exists(cat1):
    os.mkdir(cat1)

if not os.path.exists(cat2):
    os.mkdir(cat2)
#creer la masque
seuil = 35
t = m_topHat < seuil
t2 = np.prod(t, axis=2)
r_seuil = np.where(t2, 0, 255)
afficher_image(r_seuil)
cv2.imwrite(os.path.join(cat1,'figmasque.png'),r_seuil)
#plt.show()

#couper l'image masque en 50 imagettes environs
tit = image_slicer.slice(os.path.join(cat1,'figmasque.png'),50,save=False)
image_slicer.save_tiles(tit,directory=cat2)

imP = []

#mesurer le pourcentage du pixel blanc dans chaque imagette
l=0
for img in glob.glob(os.path.join(cat2,'*.png')):
    imm = cv2.imread(img)
    imPourcent = len(np.where(imm==255)[0])/(float(imm.size))
    imPourcent = np.float16(imPourcent)
    imP.append(imPourcent)


#parcourir la video chaque 3s(50 frames environs)

for j in range(0,length/50):

    if j==39:
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,1960)
        success,im = vidcap.read()
        fra=1960
    else:
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 50*j+7)
        success, im = vidcap.read()
        fra = j*50+7
    afficher_image(im)
    newpath = 'Figures/ChercherTexte/Figure%s'%fra
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(os.path.join(newpath, 'fig%s.jpg' % fra), im)
    # crop images
    titres = image_slicer.slice(os.path.join(newpath, 'fig%s.jpg'%fra), 50, save=False)
    image_slicer.save_tiles(titres, directory=newpath)
    newnewpath = 'Figures/ChercherTexte/Figures%sRenommer'%fra
    if not os.path.exists(newnewpath):
        os.makedirs(newnewpath)
    col = 0
    tmp = 0
    l   = 0
    for img in glob.glob(os.path.join(newpath, '*.png')):
        imf = cv2.imread(img)
        lig = tmp/8
        pos = [(imf.shape[0]-1)*lig,(imf.shape[1]-1)*col,3]
        if col>0 and col%7==0:
            col=0
        else:
            col+=1

        tmp+=1
        cv2.imwrite(os.path.join(newnewpath,"colo_"+str(fra)+"_"+str(pos)+"_"+str(imP[l])+".png"),imf)
        l+=1
    cv2.imwrite(os.path.join(newnewpath,"frame%s.png"%fra),im)
    shutil.rmtree(newpath)




