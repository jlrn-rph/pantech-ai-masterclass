# LABEL READING USING TESSERACT OCR

# error checking for Image module
try:
    from PIL import Image
except ImportError:
    import Image 

# import libraries 
import pytesseract
import cv2

# initialize and read camera
cam = cv2.VideoCapture(1)
while True:
    _, frame = cam.read()
    cv2.imshow('label reading', frame) # display image

    # close window when escape key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        cv2.imwrite('camera_capture.png', frame) # save image
        break

cam.release()
cv2.destroyAllWindows()

# recognize text function
def recText(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    return text

 # obtain path to tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# read text from image
info = recText('camera_capture.png') 
print(info)

# save recognized text from image to txt file
file = open('deep-learning/label-reading/result1.txt', 'w')
file.write(info)
file.close()
print('Recognized text succesfully')