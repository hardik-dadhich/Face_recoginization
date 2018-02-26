import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
id = input('enter user id')
sampleNumber = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in faces:
        sampleNumber+= 1
        
        cv2.imwrite("dataset/user." + str(id) + "." +str(sampleNumber)+ ".jpg", gray[x:x+w, y:y+h])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.waitKey(100)
        cv2.imshow('faces_here', img) 
        
    
    cv2.waitKey(1)
    if (sampleNumber > 25):
        break;

cap.release()
cv2.destroyAllWindows()

    
