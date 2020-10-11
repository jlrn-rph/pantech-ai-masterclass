# FACE DETECTION USING HAAR CASCADE FRONTALFACE ALGORITHM

# import libraries
import cv2
import os

detect = 'Person detected'
not_detect = 'Person not detected'

# initialize haar  cascade classifier
algo = 'assets/haarcascade/haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(algo)

# initialize and read camera
cam = cv2.VideoCapture(1) # external camera
while True: # capture 30 images
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

    # identify if there is a person or not
    if len(face) != 0:
        cv2.putText(img, detect, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        cv2.putText(img, not_detect, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # draw rectangle around face
    for (x, y, width, height) in face:
        cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
       
    cv2.imshow('face detection', img)

    # close window with escape key
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()