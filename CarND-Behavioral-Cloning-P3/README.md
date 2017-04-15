**Behavioral Cloning Project**

[//]: # (Image References)

[image2]: ./examples/center.jpg "Center Driving"
[image3]: ./examples/recov_1.jpg "Recovery Image 1"
[image4]: ./examples/recov_2.jpg "Recovery Image 2"
[image5]: ./examples/recov_3.jpg "Recovery Image 3"
[image6]: ./examples/recov_1.jpg "Normal Image"
[image7]: ./examples/flipped_image.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md or README.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture

I tried 2 different network architectures: LeNet (model.py lines 86-99) and Nvidia (model.py lines 101-116). The Nvidia architecture trained faster, and performed much better in autonomous mode. 

This network has a layer of normalization, followed by 5 layers of convolution (with RELU activation) and then 3 densely connected layers. The convolutional layers are 3 strided convolutions (2x2 stride, 5x5 filter), followed by 2 non-strided convolutional layers (3x3 filter). They have depths between 24 and 64 and are followed by 3 densely connected layers.

#### 2. Attempts to reduce overfitting in the model

I found that adding dropout layers to this network architecture resulted in poorer driving performance on this track. 

To avoid overfitting, i focused on data collection: acquiring different data from different directions/positions in a track, as well as driving on both tracks. Of course, i also ensured that the model was trained and validated on different datasets (model.py line 34)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115). I empirically decided to do training for 5 epochs (model.py line 126) because validation error reached a minimum at this point. Training for longer either kept validation error the same or increased it. 

#### 4. Appropriate training data

Training images were collected from both tracks (mostly from the 1st track). More details below on the different approaches to driving around the track. I used image data from all the cameras (left, center, and right), and used cropping and normalization to preprocess the images, and data augmentation (rotations) to add to the dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy for finding a working model architecture was to try some popular architectures and compare their efficiacy, with respect to rmse and driving performance, on the same set of training data. Specifically, I tried LeNet and the Nvidia architecutures. I picked these 2 because LeNet performs well in image classificaton tasks, and the Nvidia architecture has shown success in this exact task (in their End to End DL paper). 

I went straight to ConvNets, as opposed to trying more traditional ML techniques because of the nature of the problem, and the success that ConvNets have had on image data. In driving, the location of objects and their proximities to each other are extremely relevant pieces of information, and so the focus was on training a model with many layers of convolution. And in fact, the Nvidia network, which has 3 more layers of convolutional than LeNet, seemed to abstract out the right kind of information to predict the right steering angle.

After collecting ~10k images of driving data (from laps on both tracks), 2 tweaks were made to the preprocessing steps of the Nvidia network that improved the rmse - adding a lambda layer to normalize images and adding a cropping layer to remove noise from the images (background trees, etc), and allow the model to generalize more effectively. The cropping layer was especially nice because it allowed training to progress much faster, and also improved driving performance!

Finally, more images were collected to help the model deal with tricky situations (bridges, corners with dirt, and transitioning from parts of the track with sidegaurds to parts of the track without sidegaurds), and augmentation (with steering correction) was used to boost the training dataset.

Both networks were applied to the training data, and learning epochs were tuned to stop training when validation error didn't fall for each network. The Nvidia network achieved lower mse, and was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The model architecture:

- Normalization layer
- Cropping Layer: 70 pixels wide, 25 pixels high from the origin of the image.
- 5 layers of convolution (all with relu activation):
        * 2x2 Stride, depth 24, 5x5 filter
        * 2x2 Stride, depth 36, 5x5 filter
        * 2x2 Stride, depth 48, 5x5 filter
        * No stride, depth 64, 3x3 filter
        * No stride, depth 64, 3x3 filter
- 3 densely connected layers
    + 100 nodes
    + 50 nodes
    + 10 nodes
- Output layer (single node)

#### 3. Creation of the Training Set & Training Process

I used a variety of approaches to collect training data. 

**Center lane driving** Our goal is to teach the car to driving in the model. So, a few laps are driven in the middle of the track to collect this data.

![alt text][image2]

**Recovery Driving** We want to teach the car how to come back from the sides of the track. So, I collected data by allowing the car to drift and recording when the car drives back to the middle. Here's an example of a car recovering from the right side of the track to the middle.

![alt text][image3]
![alt text][image4]
![alt text][image5]

**Data Augmentation** 

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

To summarize: 

After the collection process, I had ~9000 center camera images. After using left/right cameras (with offsets to correct steering angles), my training dataset was up to 27,000 images. Then, each image was flipped horizontally. So, this double the training data size to 54,000 images. I finally randomly shuffled the data set and put 20% of the data into a validation set. So, a little over 43,000 images were used for training, and ~11,000 images were used for validation. The ideal number of epochs was 5, since validation error did not decrease after this point. I used an adam optimizer so that manually training the learning rate wasn't necessary.
