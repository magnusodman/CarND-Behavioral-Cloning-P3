import csv

from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

import numpy as np

"""
Load model loads data from disk into memory and returns two numpy arrays containing images and angles.
"""
def load_data(data_paths):
    images = []
    measurements = []

    for data_path in data_paths:
        lines = []
        with open("/".join([data_path, "driving_log.csv"])) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        

        for line in lines:
            for index in range(3):
                source_path = line[index]
                filename = source_path.split("/")[-1]
                current_path = "/".join([data_path,"IMG",filename])
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

    return np.array(images), np.array(measurements)

"""
create model sets up the nvidia inspired network fronted with a normalization and mean centering.
"""
def create_model():
    model = Sequential()
    #Normalize and mean centering
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    #nvidia Arch
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

"""
Main
Fitting model to test data and save result to file.
"""
if __name__ == "__main__":
    X_train, y_train = load_data(['./data',"./data2"])
    model = create_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save('model.h5')