# IMAGE CLASSSIFICATION USING CONVOLUTIONAL NEURAL NETWORK - TRAINING THE MODEL

# import libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# initialize variables 
batch_size = 10

# create the model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# compile the model
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

# preprocess training images by adding data augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 8.2,
                                zoom_range = 8.2,
                                horizontal_flip = 'True')

# preprocess validation images
validation_datagen = ImageDataGenerator(rescale = 1./255)

# flow training images in a specified batch size using train_datagen generator
training_dataset = train_datagen.flow_from_directory('deep-learning/image-classification/dataset/train',
                                                    target_size = (64, 64), 
                                                    batch_size = batch_size,
                                                    class_mode = 'binary')
# print(training_dataset.class_indices)

# flow validatipon images in a specified batch size using validation_datagen generator
validation_dataset = validation_datagen.flow_from_directory('deep-learning/image-classification/dataset/val',
                                                    target_size = (64, 64),
                                                    batch_size = batch_size,
                                                    class_mode = 'binary')

# train the model
model.fit_generator(training_dataset,
                    steps_per_epoch = len(training_dataset), 
                    epochs = 50,
                    validation_data = validation_dataset,
                    validation_steps = len(validation_dataset))

# save the model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')
print('Saved model to disk')
