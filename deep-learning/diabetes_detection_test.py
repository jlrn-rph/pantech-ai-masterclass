# BASIC DIABETES DETECTION USING NEURAL NETWORK - TESTING THE MODEL

# import libraries
from numpy import loadtxt
from keras.models import model_from_json

# load dataset
dataset = loadtxt('dataset/pima-indians-diabetes.csv', delimiter = ',')
x = dataset[:, 0:8] # features 
y = dataset[:, 8] # labels

# open the model
json_file  = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load model from disk
model = model_from_json(loaded_model_json)
model.load_weights('model_weights.h5')
print('Loaded model from disk')

# get predictions
predictions = model.predict_classes(x)
for i in range(5, 15):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))