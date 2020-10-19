# REAL-TIME OBJECT RECOGNITION USING MOBILENETSSD

# import libraries
import numpy as np 
import time
import cv2 
import imutils

# initialize variables
prototxt = 'assets/mobilenetssd/MobileNetSSD_deploy.prototxt.txt'
model = 'assets/mobilenetssd/MobileNetSSD_deploy.caffemodel'
confThresh = 0.2 #confidence level

# initialize classes
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# choose random colors based on length of classes
COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

# load model
print('Loading model...')
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print('Model loaded')

# initialize and read camera 
print('Starting camera feed...')
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    # resize the frame of the camera
    frame = imutils.resize(frame, width = 500)
    # will be used in plotting the bounding box and name
    (height, width) = frame.shape[:2]
    # resize the image to match the model's image size
    imgResizeBlob = cv2.resize(frame, (300, 300))
    # convert the image to blob
    blob = cv2.dnn.blobFromImage(imgResizeBlob, 0.007843, (300, 300), 127.5)
    # pass the blob as input
    net.setInput(blob)
    # proceed the image for classification
    detections = net.forward()
    # get shape
    detectionShape = detections.shape[2] 
    for i in np.arange(0, detectionShape):
        confidence = detections[0, 0, i, 2] # identify confidence level 
        # print('Confidence:', confidence)
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            print('ClassID:', idx)
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height]) # bounding box
            # print('Box Coordinates', box)
            (startX, startY, endX, endY) = box.astype("int") # convert to integer
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100) # display label with confidence level
            # draw rectangle around the object
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2) 
            # condition if 'bottle' is detected
            if CLASSES[idx] == 'bottle':
                label = "{}: {:.2f}% I need water".format(CLASSES[idx],
                confidence * 100)
            # print name above the rectangle
            if startY - 15 > 15:
                y = startY - 15
            else:
                startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # display output
    cv2.imshow("object recognition", frame)

    # close window with escape key
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()