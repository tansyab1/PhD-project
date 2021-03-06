#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing all required libraries


# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[2]:


import tensorflow as tf
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[4]:


import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# In[5]:


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[6]:


tf.__version__


# In[7]:


#Train and test data folder

train_data_dir = "/home/nguyentansy/PhD-work/Datasets/Image - Split 0-1/1"
test_data_dir = "/home/nguyentansy/PhD-work/Datasets/Image - Split 0-1/0"


# In[8]:


train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = pathlib.Path(test_data_dir)


# In[9]:


#count how many images are there
image_count = len(list(train_data_dir.glob('*/*.jpg')))
image_count


# In[10]:


total_train = len(list(train_data_dir.glob('*/*.jpg')))
total_val = len(list(test_data_dir.glob('*/*.jpg')))


# In[11]:


#get the class names
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES


# In[12]:


#Define parameter for training
batch_size = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/batch_size)
epochs = 8
num_classes = len(CLASS_NAMES) #23


# In[13]:


#We use image data generators to load the images and prepare them for the training

train_image_generator = ImageDataGenerator() # Generator for our training data
validation_image_generator = ImageDataGenerator() # Generator for our validation data


train_data_gen = train_image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode='categorical'
                                                          )

val_data_gen = validation_image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical',
                                                     classes = list(CLASS_NAMES)
                                                             )
#get class order from directories
print(train_data_gen.class_indices.keys())
print(val_data_gen.class_indices.keys())


# In[14]:


IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# base model from the pre-trained model. Resnet 50 in this case
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False 


# In[15]:


#add new classification layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(num_classes,activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


#fit the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[23]:


#create training plots
history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(30, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[24]:


base_model.trainable = True #now we want to train the base model


# In[25]:


# How many layers are in the base model
print("Layers base model: ", len(base_model.layers))

# Fine tune from layer x
fine_tune_at = 100

# Freeze all the layers before the fine tune starting layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[26]:


model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])


# In[27]:


model.summary()


# In[28]:


#Fine tune step
initial_epochs = 7
fine_tune_epochs = 3
total_epochs =  initial_epochs + fine_tune_epochs
train_batches = total_train // batch_size
print(total_val // batch_size)
validation_batches = total_val // batch_size

history_fine = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=total_epochs,
    initial_epoch = history.epoch[-1],
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[29]:


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# In[30]:


#Plot fine tuning 
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[31]:


#model save and load
import os


# In[32]:


#some time stamp 
from datetime import datetime
# current date and time.
now = datetime.now()
timestamp = datetime.timestamp(now)
print("timestamp =", timestamp)


# In[35]:


model_filename = str(timestamp)+'mymodel.h5'
model.save(model_filename)


# In[36]:


#To apply the model on new data
new_model = tf.keras.models.load_model(model_filename)

# Show the model architecture
new_model.summary()


# In[38]:


from tensorflow.keras.preprocessing import image

#image directory containing images to test
img_dir="/home/nguyentansy/PhD-work/Datasets/all/0/polyps"

for i,img in enumerate(os.listdir(img_dir)):
  tmpimage = image.load_img(os.path.join(img_dir,img), target_size=(IMG_SIZE,IMG_SIZE))   
  tmpimage = np.expand_dims(tmpimage, axis=0).astype('float32')    
  result_class=new_model.predict(tmpimage)
  print(img,";",CLASS_NAMES[result_class.argmax(axis=-1)])


# In[ ]:




