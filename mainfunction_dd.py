import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import imutils 

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

eye_model = load_model('models/cnn_eye.h5')
yawn_model=load_model('models/cnn_yawnmodel_acc.h5')

def classifier(frame,object,i,model):
    predict_r=0
    for (x,y,w,h) in object:
        pred_obj=frame[y:y+h,x:x+w]
        pred_obj = cv2.cvtColor(pred_obj,cv2.COLOR_BGR2GRAY)
        pred_obj = cv2.resize(pred_obj,(i,i))
        pred_obj= pred_obj/255
        pred_obj=  pred_obj.reshape(i,i,-1)
        pred_obj = np.expand_dims(pred_obj,axis=0)
        predict_r=model.predict(pred_obj) 
    return predict_r
        
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
yawn_p=[99]
yawn_score=0

while(True):

    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    # for (x,y,w,h) in faces:
    #    cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
    
    # for (x,y,w,h) in right_eye:
    #     cv2.rectangle(frame, (x,y) , (x+w,y+h) , (10,10,10) , 1 )
    
    # for (x,y,w,h) in left_eye:
    #     cv2.rectangle(frame, (x,y) , (x+w,y+h) , (10,10,10) , 1 )

    predict_y=classifier(frame,faces,64,yawn_model)
    predict_r=classifier(frame,right_eye,24,eye_model)
    predict_l=classifier(frame,left_eye,24,eye_model)
    
    cv2.putText(frame,"yawning per:"+str(predict_y),(10,height-40), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,"right eye per:"+str(predict_r),(10,height-60), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,"left eye per:"+str(predict_l),(10,height-80), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(predict_r>0.5 and predict_l>0.5):
        score=score-1
        cv2.putText(frame,"Open ",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(predict_y>0.5):
        yawn_score=yawn_score+1
        cv2.putText(frame,"Yawning",(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        cv2.putText(frame,"Not Yawning",(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(300,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()    
        except:  # isplaying = False
            pass
    cv2.imshow('your input video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()