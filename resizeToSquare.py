import onnxruntime as rt
import cv2
import numpy as np
import os

directory = './forTest'
os.chdir('./picture')
for picName in os.listdir(directory):
    name = picName[:-4]
    ### Store 
    img = cv2.imread(directory+'/'+picName)
    width = int(img.shape[1])
    height = int(img.shape[0])
    if width<height:
        resize = img[:width,:width]
    else:
        resize = img[:height,width-height:width]
    os.chdir('./forTest')
    cv2.imwrite(picName,resize)
    os.chdir('../')
print('-----------------------------------------------------')