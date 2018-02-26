import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("reconginizer\\traningData.yml")
id = 0
name = ""
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w+5, y+h+5), (0,255,0), 3)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if id == 1:
            name = "HARDIK"
        elif id == 2:
            name = "PM modi"
        elif id == 3:
            name = "Unknown"
        cv2.putText(img, name, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,244,5), 1)
        
            
        
    cv2.imshow('faces_herw', img)
    if cv2.waitKey() == 13:
        break;

cap.release()
cv2.destroyAllWindows()

    
