import cv2
import numpy as np

#179,213,4
#70, 90, 84
lowerBound = np.array([0,255,0]) #Change these HSV values to your own color
upperBound = np.array([35, 255, 255]) #Change these HSV values to your own color


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

            print(dribbleCounter)





    cv2.imshow('image', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('postPro', postPro)
    #cv2.imshow('res', res)











    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
