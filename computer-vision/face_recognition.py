# FACE RECOGNITION CLASSIFIER
# Dataset folders are not uploaded for the security of the faces of the people involved

# import libraries
import cv2, os
import numpy as np 

# initialize variables
cam = cv2.VideoCapture(0)
haar_cascade = 'assets/haarcascade/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade)
(images, labels, names, ids) = ([], [], {}, 0)
(width, height) = (130, 100)
count = 0
dataset = 'dataset'
print('Training...')

# load images from directory
for (subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[ids] = subdir
        subject_path = os.path.join(dataset, subdir)
        for filename in os.listdir(subject_path):
            path = subject_path + '/' + filename
            label = ids
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        ids += 1

# list the images and the labels
(images, labels) = [np.array(img_list) for img_list in [images, labels]]

# create a model of classifiers
model = cv2.face.LBPHFaceRecognizer_create()
# model = cv2.face.FisherFaceRecognizer_create()

# train the model
model.train(images, labels)

# use mobile as a webcam using IP Webcam
address = 'https://192.168.1.4:8080/video' # replace with your network's IP address
cam.open(address)

# initialize and read camera
while True:
    _, frame = cam.read()
    # convert to grayscale
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # obtain face coordinates
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = grayImg[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        # obtain prediction
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # check prediction
        if prediction[1] < 800:
           cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255))
           print(names[prediction[0]])
           count = 0 
        else:
            count += 1
            cv2.putText(frame,'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255))
            if (count > 100):
                print('Unknown person')
                cv2.imwrite('unknown.jpg', frame)
                count = 0

    # show window
    cv2.imshow('face recognition', frame)
    
    # close window with escape key
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()