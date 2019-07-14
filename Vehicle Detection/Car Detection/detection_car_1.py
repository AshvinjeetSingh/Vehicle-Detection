# -*- coding: utf-8 -*-

import cv2


cascade_src1 = 'C:/Users/ashvi/Desktop/car detection/classifier/cascade.xml'

video_src1 = 'C:/Users/ashvi/Desktop/video1.avi'

cascade_src2 = 'Bus_front.xml'

video_src2 = 'bus1.mp4'


cap = cv2.VideoCapture(video_src1)

car_cascade = cv2.CascadeClassifier(cascade_src1)


while True:
    ret, img = cap.read()
   
    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    


    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        label="car"
        cv2.putText(img,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)    
    cv2.imshow('video', img)
   
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
