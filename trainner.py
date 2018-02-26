import os
import cv2
from PIL import Image
import numpy as np

recoginizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def getImagewith(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagepath in imagepaths:
        #convert the imagepath into list redable formate
        faceImg = Image.open(imagepath).convert('L')
        #convert the faceImg into numpy array
        faceNp = np.array(faceImg, 'uint8')
        #get the id of img using splitting the path
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
        #let appeend the numpy array of img and id into list
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('traning', faceNp)
        cv2.waitKey(0)
    return np.array(IDs), faces     
        
        

        
Ids,faces = getImagewith(path)
recoginizer.train(faces, Ids)
recoginizer.save('reconginizer/traningData.yml')
cv2.destroyAllWindows()
