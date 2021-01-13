import onnxruntime as rt
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceEmbeddingExtractor import faceEmbeddingExtractor
detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=0.2)
embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')

### Store 
inputFrame = cv2.imread('./picture/anawat.jpg')
faceBoxes, faceLandmarks, alignedFaces = detector.detect(inputFrame)
extractEmbedding = embeddingExtractor.extract(alignedFaces)
embed = np.load('./storeEmbedding/embedding.npy', allow_pickle=True)
# embed = []
embed = np.append(embed, [extractEmbedding])
np.save('./storeEmbedding/embedding.npy', embed)
name = np.load('./storeEmbedding/name.npy', allow_pickle=True)
# name = []
name = np.append(name, ['Anawat'])
np.save('./storeEmbedding/name.npy', name)
