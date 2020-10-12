# OBJECT TRACKING WITH X DIRECTION BASED ON COLOR

# import libraries
import cv2
import imutils

rad_info = ''

# initialize lower and upper values of the object
lower = (1, 0, 0)
upper = (11, 251, 255)

# initialize and read camera
cam = cv2.VideoCapture(1) # external camera
while True:
    _, frame = cam.read()
    # resize frame
    frame = imutils.resize(frame, width = 600)
    # smoothen image with Gaussian blur
    blurImg = cv2.GaussianBlur(frame, (11, 11), 0)
    # convert the blurred image to HSV
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    # get the HSV image's mask to extract color area
    mask = cv2.inRange(hsvImg, lower, upper)
    # erode and dilate image to remove holes and/or noise
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    # identify boundaries to connect to nearby pixels with contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # find the center
    center = None
    # identify contour values
    if len(cnts) > 0:
        # get the maximum contour area
        c = max(cnts, key = cv2.contourArea)
        # draw minimum enclosing circle around the object
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        # find the moments to identify center
        M = cv2.moments(c)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        # draw circle 
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # plot the centroid
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            print(center, radius)
            # identify the object's radius
            if radius > 250:
                print('Stop')
                rad_info = 'Stop'
            else:
                if(center[0] < 150): 
                    print('Left')
                    rad_info = 'Left'
                elif(center[0] > 450):
                    print('Right')
                    rad_info = 'Right'
                elif(center[0] < 250):
                    print('Front')
                    rad_info = 'Front'
                else:
                    print('Stop')
                    rad_info = 'Stop'
    
    # output text to frame
    cv2.putText(frame, rad_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
    # close windows
    cv2.imshow('camera feed', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()  