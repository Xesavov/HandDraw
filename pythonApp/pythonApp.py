# import cv2
# import mediapipe as mp
# import time
# cap = cv2.VideoCapture(0)

# hands = mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=4,min_detection_confidence=0.5,min_tracking_confidence=0.5)

# pTime = 0
# cTime = 0
# while True:
#    count = 0
#    success, img = cap.read()
#    results = hands.process(img)

#    if results.multi_hand_landmarks:
#        for handLms in results.multi_hand_landmarks:
#            for id, lm in enumerate(handLms.landmark):
#                h, w, c = img.shape
#                cx, cy = int(lm.x *w), int(lm.y*h)
#                cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

#            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)

#            lms = handLms.landmark
#            if (lms[8].x > lms[12].x and lms[4].x > lms[5].x) or (lms[8].x < lms[12].x and lms[4].x < lms[5].x):
#                count += 1
#            if lms[8].y < lms[7].y:
#                count += 1
#            if lms[12].y < lms[11].y:
#                count += 1
#            if lms[16].y < lms[15].y:
#                count += 1
#            if lms[20].y < lms[19].y:
#                count += 1

#    cTime = time.time()
#    fps = 1/(cTime-pTime)
#    pTime = cTime

#    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
#    cv2.putText(img,str(int(count)), (10,140), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

#    cv2.imshow("Image", img)
#    cv2.waitKey(1)

import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 15
eraserThickness = 150
########################


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()

    img = cv2.flip(img, 1)
    if not success:
        continue
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList[0]) != 0:
        xp, yp = 0, 0
        # print(lmList)
        # tip of index and middle fingers
        x1, y1 = lmList[0][8][1:]
        x2, y2 = lmList[0][12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
            # print("Selection Mode")
            # # Checking for the click
            if y1 < 125:
                if x1 < 330:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 330 < x1 < 660:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 660 < x1 < 990:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 990 < x1:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)