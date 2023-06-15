from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import numpy as np
import os
import cv2
# import PCA library
from sklearn.decomposition import PCA
# iport TSNE library
from sklearn.manifold import TSNE
# import matplotlib library
import matplotlib.pyplot as plt
from tqdm import tqdm

pathOut = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/KvasirCapsuleIQA/Reference/imgs/'
# for video in os.listdir(pathIn):
#     if video.endswith(".mp4"):
#         extractImages(pathIn + video, pathOut)


# load the model VGG16 pretrained on ImageNet
vgg16 = models.vgg16(pretrained=True)
# remove the last layer
vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])
# set model to evaluation mode
vgg16.eval()
# move model to cuda
vgg16.cuda()
pca = PCA(n_components=1024)
tsne = TSNE(n_components=2, random_state=0)

# extract features from all images from image folder


def extractFeature(pathIn):
    # get image name
    image_name = pathIn.split('/')[-1].split('.')[0]
    # load image
    img = Image.open(pathIn)
    # resize image to 224x224
    img = img.resize((224, 224))
    # convert image to tensor
    img = transforms.ToTensor()(img)
    # normalize image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )
    img = normalize(img)
    # add dimension to image
    img = img.unsqueeze(0)
    # move image to cuda
    img = Variable(img).cuda()
    # extract feature from image
    feature = vgg16(img)
    # convert feature to numpy array
    feature = feature.cpu().data.numpy()
    # print(feature.shape)
    # extract feature and reduce to 4096 dimension using using PCA
    # feature = pca.fit_transform(feature)
    # add all features to a list and return
    return feature, image_name

# function to extract features from all images in folder and use TSNE for visualization
# Path: 2023/src/extractFeature.py


def TSNEvisualiza(inpath):
    # create a list to store all features
    features = []
    # create a list to store all image names
    img_names = []
    # read all images in folder
    for img in tqdm(os.listdir(inpath)):
        # extract feature from image
        feature, name = extractFeature(inpath + img)
        # add feature to list
        features.append(feature)
        # add image name to list
        img_names.append(name)
    # convert list to numpy array
    features = np.array(features)
    # reshape array to 2D
    features = features.reshape(features.shape[0], features.shape[2])
    # use TSNE to reduce dimension to 2D
    features = tsne.fit_transform(features)
    # plot the result
    plt.figure(figsize=(10, 10))
    plt.scatter(features[:, 0], features[:, 1], c='b', marker='x')
    # set font size of x-axis and y-axis
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # put the x label and y label
    plt.xlabel('t-SNE dimension 1', fontsize=20)
    plt.ylabel('t-SNE dimension 2', fontsize=20)
    
    # for i, txt in enumerate(img_names):
    #     plt.annotate(txt, (features[i, 0], features[i, 1]))
    # save plot to file
    # save to .eps file with minimum boarder size (bbox_inches='tight')
    plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/tsne.eps',
                format='eps', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    TSNEvisualiza(pathOut)
