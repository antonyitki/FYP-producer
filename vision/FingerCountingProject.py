# https://www.youtube.com/watch?v=p5Z_GGRCI5s&t=13s
# this code is for right hand only
# Initial code adapted to me. Volume management with fingers, https://www.youtube.com/watch?v=9iEPzbG-xLE&list=PLMoSUbG1Q_r8jFS04rot-3NzidnV54Z2q
# Added code to manage volume in between ######      #######, https://www.youtube.com/watch?v=9iEPzbG-xLE&list=PLMoSUbG1Q_r8jFS04rot-3NzidnV54Z2q&index=1
# VolumeHandControl.py


from email.mime import image
import cv2
import time
import os
import HandTrackingModule as htm
###############################################################################
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange() # -65.25 to 0.0
print(volume.GetVolumeRange())
pTime = 0
minVol = volRange[0]
maxVol = volRange[1]
# vol = np.interp(length, [50, 300], minVol, maxVol)
vol = 0
volBar = 425 # 0 for the variable volume bar
###############################################################################
wCam, hCam = 840, 680
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = "code\\Vision\\HandTracking\\Volume" #Vision\HandTracking\Volume, code\Vision\HandTracking\Volume
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    #print(f"{folderPath}/{imPath}")
    overlayList.append(image)
print(len(overlayList))
pTime = 0
detector = htm.handDetector(detectionCon = 0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = True)
    #print(lmList)
    if len(lmList) != 0:
        fingers = []
        # Big finger (Thumb)
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1,5): 
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
###############################################################################
        if totalFingers == 0:
            volume.SetMasterVolumeLevel(-65.25, None)
            volBar = 425
        if totalFingers == 1:
            volume.SetMasterVolumeLevel(-35.0, None)
            volBar = 390
        if totalFingers == 2:
            volume.SetMasterVolumeLevel(-24.0, None)
            volBar = 340
        if totalFingers == 3:
            volume.SetMasterVolumeLevel(-10.0, None)
            volBar = 300
        if totalFingers == 4:
            volume.SetMasterVolumeLevel(-5.05, None)
            volBar = 260
        if totalFingers == 5:
            volume.SetMasterVolumeLevel(0.0, None)
            volBar = 225
###############################################################################
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]
        cv2.putText(img, str(f"# fingers: {totalFingers}"), (10, 180), cv2.FONT_HERSHEY_PLAIN, 2, (25,80,220), 3)
###############################################################################
    cv2.rectangle(img, (20,225), (50, 425), (0,255,0), 3)
    cv2.rectangle(img, (20,int(volBar)), (50, 425), (0,255,0), cv2.FILLED)
    cv2.putText(img, str("Volume"), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,200,0), 2)
###############################################################################
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(f"FPS: {int(fps)}"), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (25,100,205), 3)
    cv2.imshow("Sound MANAGEMENT Tony", img)
    cv2.waitKey(1)