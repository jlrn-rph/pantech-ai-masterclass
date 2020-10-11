# FACE DETECTION USING HAAR CASCADE FRONTALFACE ALGORITHM WITH IMAGE DATABASE

# import libraries
import cv2
import os

count = 1
(width1, height1) = (130, 100)

# create variables for database of the image
dataset = 'dataset'
name = 'champ'
path = os.path.join(dataset, name)

# check if there is an existing directory
if not os.path.isdir(path):
    os.makedirs(path)

# initialize haar  cascade classifier
algo = 'assets/haarcascade/haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(algo)

# initialize and read camera
cam = cv2.VideoCapture(1) # external camera
while count < 31: # capture 30 images
    print(count)
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

    # draw rectangle around face
    for (x, y, width, height) in face:
        cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
        # crop the face
        faceOnly = grayImg[y:y+height, x:x+width]
        # resize image
        resizeImg = cv2.resize(faceOnly, (width1, height1))
        # file name of the saved images
        cv2.imwrite('%s/%s.jpg' %(path, count), faceOnly)
        count += 1
    cv2.imshow('face detection', img)

    # close window with escape key
    key = cv2.waitKey(10)
    if key == 27:
        break

print('Image captured successfully')
cam.release()
cv2.destroyAllWindows()