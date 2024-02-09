import cv2
import mediapipe as mp
import time
import numpy as np
import os
import HAND_TRACKING_MODULE as htm


cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = htm.handDetector(detectionCon=0.85)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img= cv2.flip(img,1)

    img= detector.findHands(img,draw=False)
    lmList=detector.findPosition(img)

    if len(lmList) !=0:
        #print(lmList) 
        fingers = []
        
        if lmList[4][1] > lmList[3][1]:
              fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)    
        print(fingers)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
