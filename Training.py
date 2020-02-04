import cv2
import numpy as np

# Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# Working Pipline

# 1. Read of images from video stream
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


# Capture Camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
skip =0
datafolder = "./data/"
filename = input("Enter file name")

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue
        
    
    
    # Detect all faces
    faces  = face_cascade.detectMultiScale(frame,1.2,5)
    
    
    
    if(len(faces)==0):
        continue
        
    # Sort the faces in the current frame
    faces = sorted(faces, key= lambda f : f[2]*f[3])
        
        
        
    # Take the biggest face
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        skip+=1
        
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
        
        
        
    cv2.imshow("Frame",frame)
    cv2.imshow("Face Selection", face_section)

          
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break

        
        

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

np.save(datafolder + filename,face_data)

        
cap.release()
cv2.destroyAllWindows()

