import csv

from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


import numpy as np

lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    for index in range(3):
        source_path = line[index]
        filename = source_path.split("/")[-1]
        current_path = './data/IMG/' + filename
        image = load_img(current_path)
        image_array = img_to_array(image)
        images.append(image_array)

        measurement = float(line[3])
        if index == 1:
            measurement = measurement + 0.2
        if index == 2:
            measurement = measurement - 0.2
        measurements.append(measurement)

        images.append(np.fliplr(image_array))
        measurements.append(-measurement)

    

def create_model():
    model = Sequential()
    #Normalize and mean centering
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    
    #Nvidia Arch
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

X_train = np.array(images)
y_train = np.array(measurements)

model = create_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')