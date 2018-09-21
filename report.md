# **Behavioral Cloning** 

## CarND - Project 3

### Steven Han

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Nvidia Neural Network"
[image2]: ./examples/image2.jpg "Center Driving"
[image3]: ./examples/image3.jpg "Recovery Maneuver 1"
[image4]: ./examples/image4.jpg "Recovery Maneuver 2"
[image5]: ./examples/image5.jpg "Recovery Maneuver 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off from a well-known model from Nvidia, and tune my model and data to get the best performance for my task.

My first step was to use a convolution neural network model similar to the Nvidia's neural network. I thought this model might be appropriate because it is very well known model to work well with tasks like this project. It is also one of the newer models that are known to give very high accuracy.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I have fewer epochs to stop training at its max accuracy. Then I added more training data by recording more scenarios and flipping the image.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track from lack of ability to handle sharp turns. To improve the driving behavior in these cases, I recorded more data with steeper steering angles at the edge of the track to encourage faster recovery when the vehicle steers towards the edge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-93) consisted of the following layers:

| Layer         		|     Description	       						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image							| 
| Convolution 2D     	| filters = 24, kernel = 5 x 5, same padding	|
| RELU					| Activation									|
| Convolution 2D     	| filters = 36, kernel = 5 x 5, same padding	|
| RELU					| Activation									|
| Convolution 2D     	| filters = 48, kernel = 5 x 5, same padding	|
| RELU					| Activation									|
| Convolution 2D     	| filters = 64, kernel = 3 x 3, same padding	|
| RELU					| Activation									|
| Convolution 2D     	| filters = 64, kernel = 3 x 3, same padding	|
| RELU					| Activation									|
| Flatten				| 												|
| Dense					| output: 100									|
| Dropout				| Probability: 0.2								|
| Normalization			| 												|
| RELU					| Activation   									|
| Dense					| output: 50									|
| Dropout				| Probability: 0.2								|
| Normalization			| 												|
| RELU					| Activation   									|
| Dense					| output: 10									|
| Dropout				| Probability: 0.2								|
| Normalization			| 												|
| RELU					| Activation   									|
| Dense					| output: 1										|

Nvidia Neural Network arch:
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to maneuver back to the center. These images show what a recovery looks like starting from edge:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would introduce more data with opposite turns. The flipping process was done while fetching the data from data directory and saved into cached matrix.

After the collection process, I had 16070 number of data points. I then preprocessed this data by cropping out useless part of the image: the sky. Since the vehicle only cares about the track to determines how to behave, I cropped out the top part of the image which doesn't have any useful information to speed up the process and save resources. Then to polish the data, I added a layer of normalization in the preprocess step.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by model's accuracy trend. The optimal epoch number is when the model stops learning for better accuracy.

Variable used for my model:

| Variable					|     Value				| 
|:-------------------------:|:---------------------:| 
| Epoch		       			| 10					| 
| Batch Size       			| 64					| 
| Validation split  		| 0.2					| 
  
Training loss: 0.0091
Validation loss: 0.0104

I used an adam optimizer so that manually training the learning rate wasn't necessary.  

Note: I made modifications to submit a smaller sized model.h5 as it was too big to submit previously. It turns out I forgot to define the strides in my Conv2D layer. Also, I added generators as previous reviewer recommended.

