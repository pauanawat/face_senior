import onnxruntime as rt
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceEmbeddingExtractor import faceEmbeddingExtractor
inputStream = cv2.VideoCapture(0)
detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=0.5)
embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')
### load data
faces = np.load('./storeEmbedding/embedding.npy', allow_pickle=True)
name = np.load('./storeEmbedding/name.npy', allow_pickle=True)
faces = faces.reshape((name.shape[0],512))
while True:
    isFrameOK, inputFrame = inputStream.read()
    if isFrameOK:
        faceBoxes, faceLandmarks, alignedFaces = detector.detect(inputFrame)
        if len(faceBoxes) > 0:
            ### Extract embedding ###
            extractEmbedding = embeddingExtractor.extract(alignedFaces)
            # Compare embedding
            distance = cdist(faces, extractEmbedding)
            distance = distance.reshape(distance.shape[1],distance.shape[0])
            ### Draw face ##
            for faceIdx, faceBox in enumerate(faceBoxes):
                x1, y1, x2, y2 = faceBox[0:4].astype(np.int)
                cv2.rectangle(inputFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ### find min distance
                dis_face = distance[faceIdx]
                # print(dis_face)
                if np.min(dis_face) < 1:
                    ### Draw owner
                    owner = name[np.where(dis_face == np.min(dis_face))[0]][0]
                    print(owner)
                    cv2.putText(inputFrame, owner, (x1,y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,255,255),2)                   
        cv2.imshow('Video Frame', inputFrame)
        if cv2.waitKey(1) == ord('q'):
            break