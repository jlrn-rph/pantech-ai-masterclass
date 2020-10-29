# DROWSINESS DETECTION USING 68-LANDMARK PREDICTOR

# import libraries
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils, dlib, cv2, winsound

# eye aspect ratio function to calculate vertical and horizontal coordinates
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) # aspect ratio for two eyes
    return ear

# initialize variables
count = 0
blinks = 0
frequency = 1500 # frequency of sounds emitted
duration = 1000 # sound duration in miliseconds
earThresh = 0.3 # threshold distance between vertical eye coordinate 
earFrame = 48 # consecutive frames for eye closure
shapePredictor = 'assets/68landmark/shape_predictor_68_face_landmarks.dat'

cam = cv2.VideoCapture(0) # initialize camera
detector = dlib.get_frontal_face_detector() # detect face coordinates
predictor = dlib.shape_predictor(shapePredictor) # load face predictor for facial landmark

# get the left and right eye coordinates
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

while True:
    _, frame = cam.read() # read camera
    frame = imutils.resize(frame, width = 450) # resize frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray

    rect = detector(gray, 0)

    # place landmarks upon face detection
    for r in rect:
        shape = predictor(gray, r) # place the 68-landmarks
        shape = face_utils.shape_to_np(shape) # convert coordinates to array

        # coordinates
        leftEye = shape[leftStart:leftEnd]
        rightEye = shape[rightStart:rightEnd]
        leftEar = eyeAspectRatio(leftEye)
        rightEar = eyeAspectRatio(rightEye)

        ear = (leftEar + rightEar) / 2.0

        # get eye shape and connect the contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # drowsiness detection
        if ear < earThresh:
            count += 1 # increment count if aspect ratio is less than the threshold
            if count >= earFrame: # drowsiness detected when count is greater than or equal to aspect ratio frame
                cv2.putText(frame, 'DROWSINESS DETECTED!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
                blinks = 0 # reset blinks to 0
            elif count >=1 & count < earFrame:
                blinks += 1 # increment blinks if count is greater or equal to 1 and less than the aspect ratio frame
        else:
            count = 0 # reset count to 0

    cv2.putText(frame, "Blink Counter: {}".format(int(blinks/10)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 

    cv2.imshow('drowsiness detection', frame) # display window
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # close window when 'q' is pressed
        break

cam.release()
cv2.destroyAllWindows()
