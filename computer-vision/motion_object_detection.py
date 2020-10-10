# MOTION OBJECT DETECTION USING OPENCV

# import libraries
import cv2
import time
import imutils 

count = 0

# define the change in camera
firstFrame = None
area = 500

# initialize and read camera
cam = cv2.VideoCapture(1) # external camera
time.sleep(1)

while True:
    _, img = cam.read()
    text = 'Normal'

    # resize image
    img = imutils.resize(img, width = 500)

    # convert to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # comparing the first frame and the gaussianImg
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # threshold image 
    _, threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)

    #dilate iamge
    dilateImg = cv2.dilate(threshImg, None, iterations = 2)

    # countour to find neighborhood pixels (how much area is different)
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop to draw bounding rectangle around the defined area
    for c in cnts:
        if  cv2.contourArea(c) > area:
            continue
        (x, y, width, height) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        text = 'Moving object is detected: ' + str(count)
        count += 1
        print(text)


    # text to output in the camera feed 
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show camera
    cv2.imshow('Camera Feed', img)
    cv2.imshow('Threshold Feed', threshImg)
    cv2.imshow('Image Difference Feed', imgDiff)
    
    # if 'q' is pressed, window will close
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# release camera object
cam.release()
cv2.destroyAllWindows()