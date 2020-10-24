# IMAGE CLASSSIFICATION USING CONVOLUTIONAL NEURAL NETWORK - TESTING THE MODEL

# import libraries
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

# load the model
json_file = open('deep-learning/image-classification/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("deep-learning/image-classification/model/model_weights.h5")
print("Loaded model from disk")

# classify the image 
def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (64, 64)) # preprocess image
    test_image = image.img_to_array(test_image) # convert image to array
    test_image = np.expand_dims(test_image, axis = 0) # expand dimensions
    result = model.predict(test_image) # predict the image

    # test the model
    if result[0][0] == 1:
        prediction = 'Mabel'
    else:
        prediction = 'Dipper'
    print(prediction,img_name)

# specify path to directory where images are located
import os
path = 'C:/Users/Justin/Documents/IT/AI Masterclass/Practical Sessions/deep-learning/image-classification/dataset/test'
files = [] 

# r=root, d=directories, f = files
for root, directory, file in os.walk(path):
   for f in file:
     if '.jpeg' in f:
       files.append(os.path.join(root, f))

for f in files:
   classify(f)
   print('\n')