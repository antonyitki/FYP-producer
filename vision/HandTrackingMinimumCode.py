# https://www.youtube.com/watch?v=NZde8Xt78Iw
# minimum code to run the program
# https://github.com/Balaji-Ganesh/ComputerVisionProjects
# https://github.com/search?p=1&q=Murtaza%27s+Workshop&type=Commits


import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, max_num_hands = 2)
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
while True:
    success,img = cap.read()
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRBG)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(f"id: {id}", "\t\t", cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 18, (20, 1, 200), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (20, 1, 200), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(f"fps: {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (25,100,205), 3)
    cv2.imshow("Video", img)
    cv2.waitKey(1)