
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageCms
import os, fnmatch
from sklearn.model_selection import train_test_split


# In[3]:


input_shape = (36,36,3)
nClasses = 2
patch_dim = (10,10)
im_h = 360
im_w = 360
h = int(im_h/patch_dim[0])
w = int(im_w/patch_dim[1])


# In[4]:


model = Sequential()
model.add(Conv2D(3, (5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(3, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Conv2D(3, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(75, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(nClasses, activation='softmax'))


# In[7]:


model.summary()


# ### CREATING PATCHES AND SAVING PATCH'S NP ARRAY IN A LIST

# In[24]:


def gen_patches(dtpath):
    data = Image.open(dtpath).convert("RGB")
    
    
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")
    
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab_im_data = ImageCms.applyTransform(data, rgb2lab_transform)
    
    
    im_h , im_w = data.size
    h = int(im_h/patch_dim[0])
    w = int(im_w/patch_dim[1])
    #print(im_h,im_w,h,w)
    data_lst = []
    # c = 0
    for i in range(0,im_h,h):
        for j in range(0,im_w,w):
    #         c+=1
            box = (j,i,j+w,i+h)
            a = lab_im_data.crop(box)
            a = np.array(a)
            norm = a/255.0
            norm = norm.flatten().reshape(1,3888)
            data_lst.append(norm)
    #         plt.figure(c)
    #         plt.imshow(a)
    return data_lst


# ### CREATING LABELS FOR PATCHES BY 50% RULE

# In[25]:


def gen_labels(annpath):
    anno = Image.open(annpath)
    anno_lst = []

    for i in range(0,im_h,h):
        for j in range(0,im_w,w):
            box = (j,i,j+w,i+h)
            a = anno.crop(box)
            a = np.array(a)
            k = np.count_nonzero(a)
            if k > int((w*h)/2):       #basically greater than 50% here(648)
                lb = [0,1]
            else:
                lb = [1,0]
            anno_lst.append(lb)
    return anno_lst


# ### Fetching label path from image path

# In[26]:


def lb_pth_frm_img_pth(dtpath):
    
    pattern = os.path.splitext(os.path.basename(dtpath))[0]
    anpath = 'C:\\Users\\Shashank\\Desktop\\Endo_CNN\\Train\\annotations'
    #adding exception for normal images since we have only one annotation(complete black) for all normal training images
    
    if pattern[:6] == 'normal':
        return 'C:\\Users\\Shashank\\Desktop\\Endo_CNN\\Train\\annotations\\normalm.png'
    
    for root, dirs, files in os.walk(anpath):
        for name in files:
            r = os.path.splitext(name)[0][:-1]
            if fnmatch.fnmatch(r, pattern):
                return os.path.join(root, name)


# In[27]:


train_path = 'C:\\Users\\Shashank\\Desktop\\Endo_CNN\\Train\\data'
train_path_lst = []

for dirpath,_,filenames in os.walk(train_path):
    for f in filenames:
        train_path_lst.append(os.path.abspath(os.path.join(dirpath, f)))


# ### Creating a final patch list and a final label list

# In[28]:


final_patchlist = []
final_labellist = []

for f in train_path_lst:
    lbpath = lb_pth_frm_img_pth(f)
    data_lst = gen_patches(f)
    anno_lst = gen_labels(lbpath)
    final_patchlist.extend(data_lst)
    final_labellist.extend(anno_lst)


# In[29]:


final_patchlist = np.asarray(final_patchlist)
final_labellist = np.asarray(final_labellist)


# In[30]:


X = final_patchlist.reshape((-1,36,36,3))
Y = final_labellist
#X.shape


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[37]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[33]:


model.fit(X_train,y_train,epochs=5,batch_size=20,verbose=1,validation_data=(X_test,y_test))

