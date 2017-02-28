#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./gfx/model.png "Model Visualization"
[image2]: ./gfx/center_2017_02_26_16_52_05_385.jpg "Center image"
[image3]: ./gfx/left_2017_02_26_16_52_05_385.jpg "Left Image"
[image4]: ./gfx/right_2017_02_26_16_52_05_385.jpg "Right Image"
[image5]: ./gfx/flipped_center_2017_02_26_16_52_05_385.jpg "Flipped Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results. This document

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The preprocessing of data is contained in the keras pipeline and no additional preprocessing is needed during autonomous driving.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the nvidia architecture. The model is described in detail further down.

The model includes RELU layers to introduce nonlinearity. There are studies that indicate that ELU activation improves the learning and is at least as good as RELU. I tried ELU but for my data and model it didn't show any improvement. Using tanh or sigmoid activation was not considered.
The data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting and the number of training epochs was tuned by observing the linear improvement of training loss and validation loss. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. In addition I tested and made sure that the vehicle succeeded to finish the track in opposite direction.

I tried to add a dropout layer the see if that could prevent overfitting but it didn't improve performance, at least not on my small data set.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried several approaches but finally I succeeded with a single(!) lap of center lane driving. In the end I made it without any recovery data. My suspicion is that the left/right images were enough to find a stable solution. I doubled the training data by flipping images horizontally and inverting angles. In addition to adding more test data, the flipping solved the bias to turn left.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach


The overall strategy for deriving a model architecture was to start out with a simple architecture and gradually make it more complex as needed. This proved to be very time consuming as it was hard to distinguish if the failures were due to architecture, bugs or training data. I restarted from scratch three times and what's described below is my third and final attempt. My earlier attempts involved using Udacity test data, several keyboard controlled laps and finally a single mouse controlled lap.

My first step was to create a naive network with two convolutional layers and a singe fully connected layer. This network was capable of keeping the car on track but mistook water for road near the bridge.

My second step was to use a convolution neural network model similar to LeNet. I thought this model might be appropriate because I was familiar with it and it proved successfully in the traffic sign classification project. I managed to get low loss values for both the validation and training data sets. LeNet succeeded to get across the bridge but had a hard time to stay on track for a full lap.

My third attempt was the nvidia architecture. This time the network could control the car for a full lap. And in addition to that it succeeded to navigate the track in the opposite direction. That's equal to a similar but unseen track and proves that the network is able to generalize.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. This was a very time consuming project with many degrees of freedom. What proved successfull to me was using a small data set for fast iterations on my non-GPU setup and knowing when to stop training (number of epochs). 

####2. Final Model Architecture

The final model architecture was based on the nvidia architecture and consisted of a convolution neural network with the five convolutional layers connected to four fully connected layers. In the front of the network are two layers containing cropping and normalization/mean centering.

```python
def create_model():
    model = Sequential()
    #Normalize and mean centering
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    #nvidia Arch
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
```
Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

I tried several different approaches regarding data collection.

1. Used the provided data from Udacity
2. Capturing data using keyboard
3. Capturing data using mouse.
4. Capture several laps with center lane driving and recovery.
5. Capturing a single center lane lap using a mouse. 

In the end I used a single center line lap with data augmenting. The augmenting techniques I used ere Left/Right images with correction and image flipping.

![alt text][image2]


To augment the data sat I used the left and right images provided by the simulator. 

For the left image a correction is added to the measured angle. The idea is that the car will try to go back to the center of the lane. The right image the correction is subtracted.


![alt text][image3]
![alt text][image4]


I also flipped images and and inverted angles thinking that this would add more training data and balance out any steering angle bias. 

Here is an example of image flipping:


![alt text][image5]


```python
def load_data(data_path):
    lines = []

    with open(data_path) as csvfile:
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

    return np.array(images), np.array(measurements)
```



After the collection process, I had 1319 number of data points. 
Adding the left/right images got it to 3957.
Flipping the images got it to 7914 images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. I tried to use a generator during training but it proved to be inefficient, on my non-GPU setup, to repeatedly load and parse images from disk. I discarded that solution.  

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 by observing the linear behavior of loss and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
