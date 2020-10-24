# HAND GESTURE RECOGNITION - TRAINING THE MODEL

# import libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam

# initialize variables
batch_size = 10

# create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (256, 256, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units = 150, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(units = 6, activation = 'softmax'))

# compile the model
model.compile(optimizer = Adam(learning_rate = 0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

# preprocess training images by adding data augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                rotation_range = 12.,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                zoom_range = 0.15,
                                horizontal_flip = True)

# preprocess validation images by adding data augmentation parameters to ImageDataGenerator
validation_datagen = ImageDataGenerator(rescale = 1./255)

# flow training images in a specified batch size using train_datagen generator
training_dataset = train_datagen.flow_from_directory('deep-learning/hand-gesture-recognition/dataset/train',
                                            target_size = (256, 256),
                                            color_mode = 'grayscale',
                                            batch_size = batch_size,
                                            classes = ['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                            class_mode = 'categorical')

# flow validation images in a specified batch size using validation_datagen generator
validation_dataset = validation_datagen.flow_from_directory('deep-learning/hand-gesture-recognition/dataset/train',
                                            target_size = (256, 256),
                                            color_mode = 'grayscale',
                                            batch_size = batch_size,
                                            classes = ['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                            class_mode = 'categorical')

# early stopping and model checkpoint of the model to avoid overfitting
callback_list = [
    EarlyStopping(monitor = 'val_loss', patience = 10),
    ModelCheckpoint(filepath = 'model.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)
]

# train the model
model.fit_generator(training_dataset,
                    steps_per_epoch = (len(training_dataset)/batch_size), 
                    epochs = 50,
                    validation_data = validation_dataset,
                    validation_steps = (len(validation_dataset)/batch_size),
                    callbacks = callback_list)