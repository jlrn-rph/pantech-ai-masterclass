# HAND GESTURE RECOGNITION - TESTING THE MODEL

# import libraries
from keras.preprocessing import image
from keras.models import load_model
import numpy as np 
import os
import cv2
import operator

# load the model
model = load_model('deep-learning/hand-gesture-recognition/model/model.h5')
print('Model loaded successfully')

# classify the image
def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (256, 256),grayscale=True) # preprocess image
    test_image = image.img_to_array(test_image) # convert image to array
    test_image = np.expand_dims(test_image, axis=0) # expand dimensions
    result = model.predict(test_image) # predict the image
    # test the model
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    result = classes[max_prob - 1]
    print(img_name,result)

# specify path to directory where images are located
path = 'C:/Users/Justin/Documents/IT/AI Masterclass/Practical Sessions/deep-learning/hand-gesture-recognition/dataset/test'
files = [] 

for r, d, f in os.walk(path):
   for file in f:
     if '.png' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')