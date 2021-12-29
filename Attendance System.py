import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


#Importing images from the location
path = 'Images'

#Store all the images in one list & name in another list
images = []
classNames = []
List = os.listdir(path)
print(List)
for c in List:
    curImg = cv2.imread(f'{path}/{c}')
    images.append(curImg)
    classNames.append(os.path.splitext(c)[0])
print(classNames)

#Create encoded list for known faces 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('"%Y-%m-%d %H:%M:%S"')
            f.writelines(f'\n{name},{dtString}')
 
 
encodeListKnownPerson = findEncodings(images)
print('Encoding Complete')

#calling the webcam 
capture = cv2.VideoCapture(0)

#resizing display for better FPS 
while True:
    success, img = capture.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    
    facesCurFrame = face_recognition.face_locations(imgS,model='hog')
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    #matching faces with face_encodings
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnownPerson,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnownPerson,encodeFace)
        matchIndex = np.argmin(faceDis)

        #Determine the name of the person    
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            top_pos,left_pos,bottom_pos,right_pos = faceLoc
            top_pos, left_pos, bottom_pos, right_pos = top_pos*4,left_pos*4,bottom_pos*4,right_pos*4
            cv2.rectangle(img,(right_pos,top_pos),(left_pos,bottom_pos),(0,0,255),2)
            cv2.rectangle(img,(right_pos,bottom_pos-35),(left_pos,bottom_pos),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(right_pos+6,bottom_pos-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
 
    cv2.imshow('Webcam',img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
