import cv2
import mediapipe as mp
import time
import numpy as np
import os
import HAND_TRACKING_MODULE as htm

###################################

brushThickness = 8

eraserThickness =15

###################################








folderPath = r'D:\python\workspace_machineLearning\hand_detection\Header'
myList= os.listdir(folderPath)
print(myList)
overlayList=[]
for imPath in myList:
    image= cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255,0,255)
cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector= htm.handDetector(detectionCon=0.85)
xp,yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    #1.Import image
    success, img = cap.read()
    img= cv2.flip(img,1)
    
    #2.find landmarks
    img= detector.findHands(img,draw=True)
    lmList = detector.findPosition(img)
    
    
    if len(lmList) !=0:
        #print(lmList)
        
        x1,y1,=lmList[8][1:]  #tip of index
        x2,y2=lmList[12][1:]  #tip of middle


    #3.check which fingers are up
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        if lmList[4][1] < lmList[3][1]:
              fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)  
        print(fingers)
    #4.selection mode 2 fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0 
            
            #print("selection mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header= overlayList[0]
                    drawColor=(255,0,255)
                elif 550 < x1 < 735:
                    header= overlayList[1]
                    drawColor = (255,0,0)
                elif 800 < x1 < 950:
                    header= overlayList[2]
                    drawColor= (0,255,0)
                elif 1050 < x1 < 1200:
                    header= overlayList[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1, y1 -25),(x2, y2 + 25), drawColor, cv2.FILLED)
    #5. drawing mode Index finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 5, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            if drawColor== (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,eraserThickness)

            else:
                cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp = x1, y1
    
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY) 
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img= cv2.bitwise_or(img,imgCanvas)
    
    #seting header img
    img[0:125,0:1280] =header
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    
    #cv2.imshow("Canvas",imgCanvas)
    #cv2.imshow("inverted",imgInv)
    cv2.waitKey(1)


