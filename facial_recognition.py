import cv2
import numpy as np
import os

# Recognise Faces using KNN


# 1. Load the training data (numpy arrays of all the persons)
        # x- values are stored in the numpy arrays
        # y- values we need to assign for each person
# 2. Read a video stream 
# 3. Extract faces out of it
# 4. Use knn to find the prediction of face (int)
# 5. Map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

def dist(x1,y1):
    return np.sqrt(sum((x1-y1)**2))

def knn(X,Y,query_point, k=5):
    vals = []
    
    m = X.shape[0]
    
    for i in range(m):
        dis = dist(query_point,X[i])
        vals.append((dis,Y[i]))
        
    
    vals = sorted(vals)
    
    vals = vals[:k]
    vals = np.array(vals)
    
    counts = np.unique(vals[:,1], return_counts=True)
    predict = counts[0][counts[1].argmax()]
    
    return predict



cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


dataset_path = "./data/"

face_data = []
labels  = []

class_id = 0
names ={}

for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        #Create a mapping between class_id and name
        names[class_id] = fx[:-4]
        print("loaded" , fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        
        
        #Create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
        
        
face_dataset  = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(labels,axis =0).reshape((-1,1))




while True:
    res,frame = cap.read()
    
    if res == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame,1.2,5)
    
    if len(faces)==0:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)



        
    for face in faces:
        x,y,w, h = face
        
        
        
        offset = 10
        
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        
        if np.all(np.array(face_section.shape)):
            face_section = cv2.resize(face_section,(100,100))


            #Predict label

            out = knn(face_dataset,face_labels,face_section.flatten())


            pred_name = names[int(out)]

            cv2.putText(frame, pred_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2, cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        

        
    cv2.imshow("Faces",frame)
    

    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
