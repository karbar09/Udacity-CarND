**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/hog_subsampling.png
[image5]: ./output_images/6_image_heatmap.png
[video1]: ./project.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Read in Training data and Extract HOG features

Read in training data in the first code cell (cell 4) of the jupyter notebook. All of the `vehicle` and `non-vehicle` images were read in. Here's an example of 2 images, 1 with a `vehicle` and 1 without a `vehicle`:

![alt text][image1]

I then explored different color spaces and different `hog` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  For the image above, here is what the hog output looks like for each HOG channel under the following parameters: 

- `YUV` color space 
- `orientations=9`
- `pixels_per_cell=(8, 8)`
- `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I encapsulated training/test an SVC in cell 7, used a simple grid search over HOG parameters (cell) to try different HOG parameters and see how they affect the test accuracy. For the sake of expediency, i only allow 4 parameters to vary, and compared test results after training on 900 images and testing on 100 images:

- color space: `HSV`,`HLS`, `YCrCb`
- orientations: 6 or 9
- spatial size: (16,16) or (32,32)
- histogram bins: 16 or 32

A table is printed in cell 55 that shows different levels of accuracy for each parameter combination. The `YCrCb` colorspace seemed to have the best performance, so this was selected. Due to the small training/test sizes, i did not directly use the set of parameters that had the best results, but merged performance results with choices based on inspection ( orientations =9 based on the paper result, and (32x32) spatial size and 32 histogram bins seemed to be better). 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in cell 8 using all the data (with 10% for test). Spatial, Histogram and HOG features were provided as input to the algorithm and default parameters from sklearn were used for the SVM algorithm, and an accuracy of .992 was obtained on the test data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used HOG subsamspling with window sizes of 64 pixels and 75% overlap in the verticla/horizontal direction. See the `find_cars` function in cell 22. For speed, i only search on the scale of 1.5, as it seemed to yield better results than using 1 or 2, and also better than using both 1 and 2.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 1 scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

### Video Implementation

#### 1.Video
Here's a [link to my video result](./project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. This value removed certain false positives that occured in standalone test images. To deal with false positives in the video, i created a class in cell 14 that stored the last 10 heat maps. Each time a new frame is encountered, it is added to the queue of heat maps, the maps are summed, and the threshold is applied to filter false positives. This allows us to further remove any artifacts from the video.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and construct bounding boxes around them.  

### Here are six test images and their corresponding heatmaps. Bounding boxes are drawn over the images:

![alt text][image5]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

_Training_:

Hyperparameter optimization was very expensive, since the space of parameters is so large. I believe that a random search would've been a favorable approach to a grid search. Additionally, different sampling methods could've been tried to see if the false positive detection rate went down. As an alternative to SVM's, performing transfer learning on a trained CNN such as VGG-16 or Alexnet could yielded favorable results on this image set. Additionally, data augmentation techniques such as rotations and flips could have helped in training.

_Pipeline_:

The pipeline itself is rather wobbly and detects cars late, and sometimes even loses them in subsequent frames. Better vehicle detection could be implemented, and bounding boxes could be smoothed over multiple frames to account for the wobble.

The pipeline fails when cars are partially in the region of interest. Using more and smaller window sizes could help with detection of these vehicles. Additionally, shadows and bumps cause instability in the bounding boxes and sometimes even lose detected vehicles.

