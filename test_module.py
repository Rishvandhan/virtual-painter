import HAND_TRACKING_MODULE as htm
import cv2
import mediapipe as mp
import time



pTime   = 0
cTime   = 0

cam = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success,img=cam.read()
    img =detector.findHands(img,False)
    lmList=detector.findPosition(img)
    if len(lmList) != 0: 
        print(lmList[4])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
    cv2.imshow("test",img)
    
    cv2.waitKey(1)