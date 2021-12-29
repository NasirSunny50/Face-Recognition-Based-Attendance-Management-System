import dlib
import cv2
import face_recognition
import numpy as np


#Load Image From Path
imgMain = face_recognition.load_image_file('images/sunny.jpg')
imgMain = cv2.cvtColor(imgMain,cv2.COLOR_BGR2RGB)
imgCompare = face_recognition.load_image_file('images/obama.jpg')
imgCompare = cv2.cvtColor(imgCompare,cv2.COLOR_BGR2RGB)

#Find Face from Primary Image
faceLocate = face_recognition.face_locations(imgMain)[0]
EncodeMain = face_recognition.face_encodings(imgMain)[0]
#Draw Rectangle on the Face
cv2.rectangle(imgMain,(faceLocate[3],faceLocate[0]),(faceLocate[1],faceLocate[2]),(0,0,255),2)


#Find Face from Primary Image
faceLocate = face_recognition.face_locations(imgCompare)[0]
EncodeCompare = face_recognition.face_encodings(imgCompare)[0]
#Draw Rectangle on the Face
cv2.rectangle(imgCompare,(faceLocate[3],faceLocate[0]),(faceLocate[1],faceLocate[2]),(0,0,255),2)

#Showing Results
results = face_recognition.compare_faces([EncodeMain],EncodeCompare)
print(results)


cv2.imshow('Main', imgMain)
cv2.imshow('Compare', imgCompare)
cv2.waitKey(0)


