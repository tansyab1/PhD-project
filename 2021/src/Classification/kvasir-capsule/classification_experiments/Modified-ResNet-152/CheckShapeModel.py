#  file to check shape of the model
import torch
import os 
import torchvision.models as models


if __name__ == '__main__':
    model = models.resnet152(pretrained=True)
    print(model)