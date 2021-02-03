import onnxruntime as rt
import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceEmbeddingExtractor import faceEmbeddingExtractor
detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=1)
embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')

directory = './picture/forTest'
count = True
for picName in os.listdir(directory):
    name = picName[:-4] 
    # print(name)
    ### Store 
    img = cv2.imread(directory+'/'+picName)
    faceBoxes, faceLandmarks, alignedFaces = detector.detect(img)
    extractEmbedding = embeddingExtractor.extract(alignedFaces)
    for faceIdx, faceBox in enumerate(faceBoxes):
        x1, y1, x2, y2 = faceBox[0:4].astype(np.int)
        # print(x1,x2,y1,y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        ### gt is ground truth
        if count :
            gt = []
        else:
            gt = np.load('./gt.npy', allow_pickle=True)         
        gt = np.append(gt, [x1,y1,x2,y2])
        np.save('./gt.npy', gt)
        count = False
        os.chdir('./picture/GT_forTest')
        cv2.imwrite(picName,img)
        os.chdir('../../')

