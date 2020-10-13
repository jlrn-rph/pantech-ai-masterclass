# FACE EMOTION RECOGNITION USING FACIAL EMOTION RECOGNITION LIBRARY

# import libraries
import cv2 as cv
from facial_emotion_recognition import EmotionRecognition 

# initialize the EmotionRecognition with cpu
emotion_recognition =  EmotionRecognition(device = 'cpu')

# initialize and read camera
cam = cv.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = emotion_recognition.recognise_emotion(frame, return_type = 'BGR')
    # show window
    cv.imshow ('emotion recognition', frame)
    # close window with escape key
    key = cv.waitKey(1)
    if key == 27:
        break

cam.release()
cv.destroyAllWindows()