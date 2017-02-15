#**Traffic Sign Recognition** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator or the Udacity's provided dataset to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity's provided simulator and my drive.py file, the car can be driven autonomously around the track by executing: 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

I applied the NVIDIA model from [this paper](https://arxiv.org/pdf/1604.07316.pdf)

![NVIDIA Model layout](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. One thousand images were randomly chosen for validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I used an adam optimizer, so the learning rate was not tuned manually. Everything else was chosen by trial and error, including steering angles correction for left and right images, batch sizes, number of epochs, color space, samples_per_epoch, nb_val_samples, etc...

After trying some datasets created on my own to no good I used Udacity's provided data.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network similar to the NVIDIA model. I thought this model might be appropriate because  it was created for a real life driving environment and should generalize well for the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by randomly choosing one thousand images for validation and everything else for training. I started testing the model with 10, 15 and 20 epochs and I found out that the sweet spot for this initial model was in epoch 12 approximately. I started using the three channels of the RGB color space and after some trial and error I produced a model that could drive around the first track, but I couldn't reproduce it. Possibly a lucky initialization that didn't happen a second time. I gave up the project for some time and continued to the Advanced Lane Finding lesson. In this one I learned that the saturation channel works best to identify lane lines and I decided to try using it in this project. This was the silver bullet.

I was able to produce a model that can drive the car in both tracks after training just one epoch. This is the model that I'm submitting.

####2. Creation of the Training Set & Training Process

By using Udacity's dataset I had ~24000 data points available.

As part of my augmentation process I flip images and angles in order to rebalance the dataset which is biased to the left. When I started working in my first models I also augmented the brightness, but in the late models, as I chose to use just the S channel of the HSV color space this was no longer necessary. Images are also cropped to remove some of the sky and the hood.

I set the number of samples per epoch to 24000, which is aproximately the number of different images I have, including left and right images. Skipping the images saved for validation, the training generator will randomly spit images from the remaining ~21000, half of them will be flipped (the number of different images that can be generated this way is ~42000, including flipped images). The validation set helped determine if the model was over or under fitting, but as I said, one epoch was enough. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####Conclussion
Although for a few days, after weeks of trying, I was about to give up and submit my initial model with a .h5 file that I generated just out of luck, I finally found that using the S channel from the HSV color space helped me creating a much better model. This project was really challenging and finally, very rewarding.