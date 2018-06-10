[//]: # (Image References)

[track1]: ./img/track1.png "Simulator Track 1"
[track2]: ./img/track2.png "Simulator Track 2"
[track2_small]: ./img/track2_2.png "Simulator Track2 small"
[nvidia]: ./img/nvidia.png "Nvidia CNN"
[example]: ./img/example_img.jpg "Recovery Image"
[angleFreq]: ./img/data_freq.jpg "Steering Angle Frequency"
[crop]: ./img/crop.png "Cropped Image"
[gamma]: ./img/gamma.png "Adjust Brightness"
[flip]: ./img/flip.png "Flipped Image"
[shear]: ./img/shear.png "Sheared Image"
[angleFreqAftAug]: ./img/data_freq_aft_aug.png "Steering Angle Frequency After Data Augmentation"
[jumpWater]: ./img/jump_water.png "Jumps into Water"

# **Behavioral Cloning** 

![track2]

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---

## Dependencies

1. Python
1. Matplotlib
1. Numpy
1. Keras
1. Tensorflow
1. Sklearn

## Getting Started

#### 1. Files used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Running the code
Using the Udacity provided simulator and `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## Model Architecture and Data Collection

#### 1. Model Architecture

I use an existing architecture from [Nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which was developed to achieve similar behavior cloning objective. 

The model is a convolution neural network that accepts a 64x200 (The original model takes in 66x200) pixel-wide image with 3 channels and outputs one node, which can be the steering angle of the vehicle. This CNN consisted of a normalization layer and 5 convolution layers followed by 4 fully connected layers. 

![nvidia]

I use includes `tan` function as activation layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Dropout layers is not added due to suggestions from the community.

#### 2. Dataset

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and recovery driving (i.e. recovering the car to the center of the road when it is off to the side). 

I created two separate datasets in attempt to overcome an issue which will be discussed later.

I started off with using [Udacity's training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) then I augment the dataset with recovery driving and more recordings in maneuvering sharp turns. The total number of images is `~27,700`, including images from all three cameras.

I also created another dataset by recording myself driving the car in the simulator in two laps, one forward and the other backward. I also augment this dataset with recovery driving. The total number of images is `~8670`, including images from all three cameras.

#### 3. Data Capturing

The simulator offers two tracks.

**Track1:**

![track1]

**Track2:**

![track2_small]

Only Track1 is used to record the data for training the model. Track2 is reserved to test how well the model generalize, and the model should have never been exposed to Track2.

During training, the simulator records images of the car driving in the track. At any point in the recording, there are three images taken from the left, center and right camera at the front of the car. Besides the images, the simulator also records the steering angle, throttle, and speed of the car, but in the interest of this project, only the steering angle is considered.

![example]


## Data Preprocessing and Augmentation

#### 1. Unbalanced Data

Due to the fact that the car is driving straight for most of the time in Track1, the training data is heavily skewed towards zero or close to zero steering angle.

![angleFreq]

To counter the frequent occurrence of low steering, I record more instances of maneuvering the car through sharp turns. However, a more effective strategy is to augment the data with new images.

#### 1a. Consider the Images from Stereo Camera

The images from the side cameras are good additions to the dataset. To account for the change in position of the camera on the car, a bias of +0.25 is added to the steering angle of left camera images (making the car tend to turn right), and -0.25 is subtracted from the steering angle of right camera images.

#### 2. Data Augmentation

Each training image is undergoing the following augmentation strategies with a probability:

- `30%` chance to adjust the **brightness** of the image (to simulate different daylight condition)
- `50%` chance to **flip** the image horizontally (to balance the bias of the dataset to turn left or right)
- `40%` chance to **shear** the image at a random location of the image (to introduce more less frequent steering angles in the dataset)

Below are shown the effect of each of the augmentation

![gamma]
![flip]
![shear]

When the image is sheared, the steering angle is also adjust with the angle of shearing.

#### 3. Preprocessing

The images in the dataset is cropped and resize to 200x64 to remove the information unrelated to the road condition. The CNN has an embedded layer that normalizes each pixel value to the range of [-0.5, 0.5].

![crop]

#### 4. Result

The frequency of steering angle in the dataset after augmentation is as such:

![angleFreqAftAug]

The introduction of the stereo camera images creates two addition peaks in the graph. However, the shearing of images lead a more gaussian like distribution of steering angles.


## Training Strategy

#### 1. Training Pipeline

For training purpose, I use AWS's g2.x2large instance with attached GPU.

Since the dataset is huge, it makes sense to utilize python's generator pattern to feed the batch of images, after preprocessing, to the model to avoiding preloading the entire dataset into a machine's memory.

Keras's `fit_generator` API is perfect with of combination of using generator.

The dataset is split 20% as validation set while the rest is used as training set.

I have created two generators for the training set and validation set. The batch size is `512`.

```
train_gen = g.sample_generator(train_samples, DATA_PATH, batch_size=BATCH_SIZE)
validation_gen = g.sample_generator(validation_samples, DATA_PATH, batch_size=BATCH_SIZE, augment_enable=False)
```

I use `Adam` optimizer with learning rate as `1e-3` and train for `12 epochs` with the number images equal to that in the dataset per epoch.

#### 2. Hyperparameter Tuning

During the entire training process, overfitting is not an issue. The validation error is well under training error, such that model seems rather to be underfitting. This is resolved by extending the number of epoch to train for. Data augmentation certainty helps to prevent overfitting.

#### 3. Training Process

At first, the cropping process is a layer within the model and is not a separate preprocessing step. For this reason the model is accepting the full image (320x160) as input. The model trained with Udacity's dataset is doing reasonable across different sections of the track. However, there are difficulties in recovery from getting too close to the sides of the bridge. This is resolved by adding more instances of maneuvering at the bridge to the dataset.

#### 3a. The One Consistent Problem

However, at some point in the process of training, the model is doing well at all sections of Track1 except the following spot in which the model decides to turn right into the water in the middle of making a left turn.

![jumpWater]

I have tried to add a recovery instance just for this particular spot into the dataset, but it is useless. At this point, the dataset is populated with recovery driving, so it could be learning to purposely go into a scenario where recovery driving is necessary instead of doing it only when is necessary.

So I decide to create a fresh dataset with two laps of training data with minimum recovery driving. However, the model is still stuck at this particular spot.

However, a closer look at this particular scene is that the lane lines disappear in a short distance ahead due to the sharp left turn. The right lane line is very close to the borders of the lake and the model could have misunderstood the border as a continuation of the lane line and thus turn right instead of left into the lake. 

With this in mind, I reuse the Udacity dataset mixed with recovery driving but crops five extra rows from the top of the image and abstract the cropping layer out of the model as an individual precessing step. This makes the model to take in (200x64) image as input.

This finally resolves the issue.

## Result

Speed is also an important factor in evaluating how the model performs. The dataset has very few instances of large turn angles and is not likely to make wide angle turns, so it is not advantageous for the model to go fast when test it on the track. 

The training speed is around 30MPH, so I tune down the speed for autonomous driving for Track1 to 20MPH, and the model is doing great. 

- [Track1](https://youtu.be/SJyCEcUxbe0): https://youtu.be/SJyCEcUxbe0

For Track2, which the model has been never exposed to during training is doing exceptional at 10MPH.

- [Track2](https://youtu.be/Utz5BzwYJ1Y): https://youtu.be/Utz5BzwYJ1Y



