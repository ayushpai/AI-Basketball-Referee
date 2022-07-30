import cv2
import numpy as np

import firebase_admin
import time
from firebase_admin import credentials
from firebase_admin import firestore

#179,213,4
#70, 90, 84
lowerBound = np.array([0,255,0])
upperBound = np.array([35, 255, 255])


#lowerBound = np.array([0,100,50])
#upperBound = np.array([50, 255, 255])

#ball detection
vid = cv2.VideoCapture(0)
topYPixel = 0
bottomYPixel = 0

leftXPixel = 0
rightXPixel = 0
bottomYList = []
dribbleCounter = 0

dThreshold = 0 # make auto adjusting based on radius of the ball (how far away it is from the camera
killThreshold = 0

cred = credentials.Certificate(r"C:\Users\ayush\PycharmProjects\BasketballRef\service.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'live').document(u'tracker')
doc = doc_ref.get()

stepList = []
dribbleList = []
localDribbles = 0
localSteps = 0
stepsSinceLastDribble = 0

travelWatcher = False

dribbleList.append(0)
dribbleList.append(0)

while True:
    # ball
    ret, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    postPro = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask=postPro)

    if len(np.where(postPro == 255)[0]) > 5:
        bottomYPixel = np.max(np.where(postPro == 255)[0])
        topYPixel = np.min(np.where(postPro == 255)[0])
        leftXPixel = np.min(np.where(postPro == 255)[1])
        rightXPixel = np.max(np.where(postPro == 255)[1])

        radiusX = (rightXPixel - leftXPixel) / 2
        radiusY = (bottomYPixel - topYPixel) / 2

        finalRadius = (radiusX + radiusY) / 2
        centerX = (leftXPixel + rightXPixel) / 2
        centerY = (topYPixel + bottomYPixel) / 2
        cv2.circle(frame, (int(centerX), int(centerY)), int(finalRadius), (0, 255, 0), thickness=2, lineType=8, shift=0)



        #print(finalRadius)

        if finalRadius > 65:
            dThreshold = 130
            killThreshold = 200
        elif 65 > finalRadius > 25:
            dThreshold = 75
            killThreshold = 250
        else:
            dThreshold = 25
            killThreshold = 300

        bottomYList.append(bottomYPixel)
        if bottomYList[len(bottomYList) - 1] > bottomYList[len(bottomYList) - 2] + dThreshold and bottomYList[len(bottomYList) - 1] > killThreshold:
            bottomYList.clear()
            dribbleCounter = dribbleCounter + 1

            #print(dribbleCounter)

    steps = doc_ref.get(field_paths={'steps'}).to_dict().get('steps')
    stepList.append(steps)
    dribbleList.append(dribbleCounter)
    #print(steps)

    if stepList[len(stepList) - 1] > stepList[len(stepList) - 2] and dribbleList[len(dribbleList) - 1] == dribbleList[len(dribbleList) - 2] and not travelWatcher:
        travelWatcher = True

    if travelWatcher and stepList[len(stepList) - 1] > stepList[len(stepList) - 2] + 1 and dribbleList[len(dribbleList) - 1] == dribbleList[len(dribbleList) - 2]:
        stepsSinceLastDribble = stepsSinceLastDribble + 1
    elif travelWatcher and stepList[len(stepList) - 1] > stepList[len(stepList) - 2] + 2 and dribbleList[len(dribbleList) - 1] == dribbleList[len(dribbleList) - 2]:
        stepsSinceLastDribble = stepsSinceLastDribble + 2


    print("Steps Since Last Dribble: ", stepsSinceLastDribble, ", Steps: ", steps, ", Dribbles: ", dribbleCounter, ", Travel Watcher: ", travelWatcher)






    cv2.imshow('image', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('postPro', postPro)
    #cv2.imshow('res', res)











    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
