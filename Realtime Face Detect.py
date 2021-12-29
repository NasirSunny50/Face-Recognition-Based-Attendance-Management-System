import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)
all_face_location = []

while True:
    ret,current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_location = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    
    for i,current_face_location in enumerate(all_face_location):
        #splitting the tupple 
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #change the position
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #print location of image
        print("Found face {} at top:{}, right:{}, bottom:{}, left:{}".format(i+1,top_pos,right_pos,bottom_pos,left_pos))
        
        #draw rectangle
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        
    #show
    cv2.imshow("Webcam Video",current_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()  
