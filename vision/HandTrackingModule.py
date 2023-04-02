# List of 21 values of each hand. Ask just for point value. Easy for projects.
# This module is to get the positions of landmarks easily
# https://www.youtube.com/watch?v=NZde8Xt78Iw
# https://github.com/Balaji-Ganesh/ComputerVisionProjects
# https://github.com/search?p=1&q=Murtaza%27s+Workshop&type=Commits


import cv2
import mediapipe as mp
import time


class handDetector():

    def __init__(self, mode = False, maxHands = 2, modelComplexity=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRBG)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    # print(id, lm)
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    #print(f"id: {id}", "\t\t", cx, cy)
                    lmList.append([id, cx, cy])
                    if draw:
                        if id == 0:
                            cv2.circle(img, (cx, cy), 17, (20, 1, 200), cv2.FILLED)
                        if id == 4:
                            cv2.circle(img, (cx, cy), 10, (20, 1, 200), cv2.FILLED)
        return lmList
          
            
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(f"fps: {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (25,100,205), 3)
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()