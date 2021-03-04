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

### load store data
faces = np.load('./storeEmbedding/embedding.npy', allow_pickle=True)
names = np.load('./storeEmbedding/name.npy', allow_pickle=True)
gt = np.load('./gt.npy', allow_pickle=True)
faces = faces.reshape((names.shape[0],512))
gt = gt.reshape((names.shape[0],4))

directory = './picture/forTest'
warnings.filterwarnings("ignore")

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


detector = faceDetectorAndAlignment('models/faceDetector.onnx', processScale=0.35)
embeddingExtractor = faceEmbeddingExtractor('models/r100-fast-dynamic.onnx')

face_scale = []

for i in range(13):
    scale = 0.7 ** (i)
    print('scale:', scale)
    count = 0
    list_face = []
    for picName in os.listdir(directory):
        name = picName[:-4]
        # print(name)
        ### Load image and resize 
        img = cv2.imread(directory+'/'+picName)
        width = int(img.shape[1]*scale)
        height = int(img.shape[0]*scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        faceBoxes, faceLandmarks, alignedFaces = detector.detect(img)
        ### resize ground truth
        if i > 0:
            gt[count] = gt[count] * 0.7
        x1A, y1A, x2A, y2A = gt[count].astype(np.int)
        if y2A-y1A < x2A-x1A:
            widthGT = int(y2A-y1A)
        else:
            widthGT = int(x2A-x1A)
        # cv2.rectangle(img, (x1A, y1A), (x2A, y2A), (0, 255, 0), 2)
        if len(faceBoxes)>0:
            for faceIdx, faceBox in enumerate(faceBoxes):
                ### detection 
                x1, y1, x2, y2 = faceBox[0:4].astype(np.int)
                iou = intersection_over_union(faceBox,gt[count])
                # print(name, ', img size:', width, 'x', height, ', face size:', x2-x1, 'x', y2-y1, ', IoU =', iou)
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
                    # print('recognition correct')
                    list_face.append((widthGT, iou, np.min(dis_face)))
                else:
                    # print('recognition fail and it recog to',owner)
                    list_face.append((widthGT, iou, 1.4))
            else:
                # print(name,'can not recog and min distance =',np.min(dis_face),'recog to',names[np.where(dis_face == np.min(dis_face))[0]][0])
                list_face.append((widthGT, iou, np.min(dis_face)))
        else:
            # print(name, ', img size:', width, 'x', height, 'can not detect')
            list_face.append((widthGT, 0, 1.4))
        count+=1
    face_scale.append(list_face)

### format data face
sizes_face = {}
size_face = []
detect_face = []
recog_face = []

for ls in face_scale:
    for data in ls:
        if data[0] in sizes_face:
            sizes_face[data[0]][1] = (sizes_face[data[0]][1] * sizes_face[data[0]][0] + data[1]) / (sizes_face[data[0]][0]+1) 
            sizes_face[data[0]][2] = (sizes_face[data[0]][2] * sizes_face[data[0]][0] + data[2]) / (sizes_face[data[0]][0]+1) 
            sizes_face[data[0]][0] += 1
        else:
            sizes_face[data[0]] = [1, data[1], data[2]]
for size_in_dict in sizes_face.keys():
    size_face.append(size_in_dict)
size_face = sorted(size_face)
for size_in_list in size_face:
    detect_face.append(sizes_face[size_in_list][1])
    recog_face.append(sizes_face[size_in_list][2])  
size_face = np.array(size_face)
detect_face = np.array(detect_face)
recog_face = np.array(recog_face)

### save to excel
# face
data = []
for i in range(len(size_face)):
    data.append([size_face[i], detect_face[i], recog_face[i]])
df = pd.DataFrame (data, columns=['size','IoU','min distance'])
df.to_excel('test0.35.xlsx')

### plot graph
graph = [detect_face, recog_face]
labels = ["IoU", "distance"]
colors = ["b", "g"]

f,axs = plt.subplots(2)

# ---- loop over axes ----
for i,ax in enumerate(axs.flat):
    x = size_face
    ax.plot(x, graph[i],color=colors[i],label=labels[i])
    ax.set_xlim(500, 0)
    ax.grid()
    ax.legend(loc="upper right")
plt.setp(axs[-1], xlabel='size (pixel)')
plt.show()