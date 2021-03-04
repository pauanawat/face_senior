import onnxruntime as rt
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.spatial.distance import cdist
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceEmbeddingExtractor import faceEmbeddingExtractor
warnings.filterwarnings("ignore")
### load data
faces = np.load('./storeEmbedding/embedding.npy', allow_pickle=True)
names = np.load('./storeEmbedding/name.npy', allow_pickle=True)
faces = faces.reshape((names.shape[0],512))
scales = [1, 0.5, 0.35]
for i in range(len(scales)):
    scale = scales[i]
    print('process scale =', scale)
    directory = './picture/forTestPixel'

    for picName in os.listdir(directory):
        name = picName[:-4]
        detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=scale)
        embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')
        ### Load image
        img = cv2.imread(directory+'/'+picName)
        width = int(img.shape[1])
        height = int(img.shape[0])
        faceBoxes, faceLandmarks, alignedFaces = detector.detect(img)
        if len(faceBoxes)>0:
            ### Extract embedding ###
            extractEmbedding = embeddingExtractor.extract(alignedFaces)
            # Compare embedding
            distance = cdist(faces, extractEmbedding)
            distance = distance.reshape(distance.shape[1],distance.shape[0])
            for faceIdx, faceBox in enumerate(faceBoxes):
                ### detection 
                x1, y1, x2, y2 = faceBox[0:4].astype(np.int)
                print(name, ', img size:', width, 'x', height, ', face size:', x2-x1, 'x', y2-y1)
                ### find min distance
                dis_face = distance[faceIdx]
                # print(dis_face)
                if np.min(dis_face) < 0.9:
                    ### print owner name
                    owner = names[np.where(dis_face == np.min(dis_face))[0]][0]
                    print('distance:', dis_face, 'recog to:', owner)
                else:
                    print('min distance:', np.min(dis_face), 'recog to:',names[np.where(dis_face == np.min(dis_face))[0]][0])
        else:
            print(name, ', img size:', width, 'x', height, 'can not detect')
