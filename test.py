import onnxruntime as rt
import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceEmbeddingExtractor import faceEmbeddingExtractor
inputStream = cv2.VideoCapture(0)
detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=1)
embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')
### load data
faces = np.load('./storeEmbedding/embedding.npy', allow_pickle=True)
names = np.load('./storeEmbedding/name.npy', allow_pickle=True)
gt = np.load('./gt.npy', allow_pickle=True)
faces = faces.reshape((names.shape[0],512))
gt = gt.reshape((names.shape[0],4))

directory = './picture/forTest'

def intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

for i in range(3):
    scale = 0.5 ** (i)
    print('scale:', scale)
    count = 0
    for picName in os.listdir(directory):
        name = picName[:-4]
        ### Store 
        img = cv2.imread(directory+'/'+picName)
        width = int(img.shape[1]*scale)
        height = int(img.shape[0]*scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        faceBoxes, faceLandmarks, alignedFaces = detector.detect(img)
        ### draw ground truth
        if i > 0:
            gt[count] = gt[count] * 0.5
        x1A, y1A, x2A, y2A = gt[count].astype(np.int)
        # cv2.rectangle(img, (x1A, y1A), (x2A, y2A), (0, 255, 0), 2)
        if len(faceBoxes)>0:
            for faceIdx, faceBox in enumerate(faceBoxes):
                ### draw detection 
                x1, y1, x2, y2 = faceBox[0:4].astype(np.int)
                print(name, ', img size:', width, 'x', height, ', face size:', x2-x1, 'x', y2-y1, ', IoU =', intersection_over_union(faceBox,gt[count]))
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            ### recogintion
            extractEmbedding = embeddingExtractor.extract(alignedFaces)
            distance = cdist(faces, extractEmbedding)
            distance = distance.reshape(distance.shape[1],distance.shape[0])
            dis_face = distance[faceIdx]
            if np.min(dis_face) < 0.9:
                ### check owner
                owner = names[np.where(dis_face == np.min(dis_face))[0]][0]
                if owner==name:
                    print('recognition correct')
                else:
                    print('recognition fail and it recog to',owner)
            else:
                print('can not recog and min distance =',dis_face)
        else:
            print(name, ', img size:', width, 'x', height, 'can not detect')
        ### save new img 
        # os.chdir('./picture/result') 
        # cv2.imwrite(name+str(scale)+'.jpg',img) 
        # os.chdir('../../') 
        count+=1
    print('-----------------------------------------------------')