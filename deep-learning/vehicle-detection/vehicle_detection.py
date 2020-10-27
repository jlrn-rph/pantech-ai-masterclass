# VEHICLE DETECTION AND TRACKING WITH TWO CAMERAS

# import libraries
import cv2, imutils

# initialize cascade files
src_cascade = 'assets/cars/cars.xml'
car_cascade = cv2.CascadeClassifier(src_cascade)

# initialize camera
cam = cv2.VideoCapture('assets/cars/traffic.mp4')
cam1 = cv2.VideoCapture(1)

while True:
    detected = 0
     # read camera
    _, img = cam.read()
    _, img1 = cam1.read()

    # for North direction
    img = imutils.resize(img, width = 300) # resize image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to gray
    cars = car_cascade.detectMultiScale(gray, 1.1, 1) # coordinates of vehicle in frame
    for (x, y, width, height) in cars:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2) # draw rectangle
    cv2.imshow('North', img)
    len_car = str(len(cars))
    int_len = int(len_car)
    detected = int(int_len)
    n = detected

    # for South direction
    img1 = imutils.resize(img1, width = 300) # resize image
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # convert image to gray
    cars1 = car_cascade.detectMultiScale(gray1, 1.1, 1) # coordinates of vehicle in frame
    for (x, y, width, height) in cars1:
        cv2.rectangle(img1, (x, y), (x + width, y + height), (255, 0, 0), 2) # draw rectangle
    cv2.imshow('South', img1)
    len_car1 = str(len(cars1))
    int_len1 = int(len_car1)
    detected1 = int(int_len1)
    n1 = detected1

    print('--------------------------------')
    print('North: %d' % (n))
    if n >= 2:
        print('North more traffic')
    else:
        print('No traffic')

    print('--------------------------------')
    print('South: %d' % (n1))
    if n >= 2:
        print('South more traffic')
    else:
        print('No traffic')

    # close window on 'escape' key
    if cv2.waitKey(30) == 27: 
        break

cam.release()
cv2.destroyAllWindows()