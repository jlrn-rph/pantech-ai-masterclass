# BASIC DIABETES DETECTION USING NEURAL NETWORK - TRAINING THE MODEL

# import libraries
from numpy import loadtxt
from keras.models import Sequential, model_from_json
from keras.layers import Dense

# load dataset
dataset = loadtxt('dataset/pima-indians-diabetes.csv', delimiter = ',') 
x = dataset[:, 0:8] # features 
y = dataset[:, 8] # labels

# build the model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
# model.summary()

# compile to optimize the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# train the model
model.fit(x, y, epochs = 5000, batch_size = 100)

# gives overall accuracy
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy * 100))

# save the model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')
print('Saved model to disk')
