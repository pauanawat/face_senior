import onnxruntime as rt
import cv2
import numpy as np
import os
import warnings
from scipy.spatial.distance import cdist
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceEmbeddingExtractor import faceEmbeddingExtractor
detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=1)
embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')
warnings.filterwarnings("ignore")
directory = './picture/store'
count = True
for picName in os.listdir(directory):
    name = picName[:-4] 
    print(name)
    ### Store 
    img = cv2.imread(directory+'/'+picName)
    faceBoxes, faceLandmarks, alignedFaces = detector.detect(img)
    extractEmbedding = embeddingExtractor.extract(alignedFaces)
    for faceIdx, faceBox in enumerate(faceBoxes):
        if faceIdx>0:
            print('has more than one face')
        x1, y1, x2, y2 = faceBox[0:4].astype(np.int)
        # print('detect crop size:',x2-x1,'x',y2-y1)
    if count :
        embed = []
        names = []
    else:
        embed = np.load('./storeEmbedding/embedding.npy', allow_pickle=True)         
        names = np.load('./storeEmbedding/name.npy', allow_pickle=True)
    embed = np.append(embed, [extractEmbedding])
    np.save('./storeEmbedding/embedding.npy', embed)

    names = np.append(names, [name])
    np.save('./storeEmbedding/name.npy', names)

    count = False
