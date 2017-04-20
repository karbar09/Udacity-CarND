**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_test1.png "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera matrix and distortion coeffcients are calculated using the `calibrate` function in cell 1 of the ipython notebook (`Advanced Line Detection.ipynb`). 

These are calculated by mapping object points `objpoints`, which are coordinates of the chessboard corners, to image points (actual positions of the corners in the image in pixels). For this board, `nx = 9`, and `ny = 6`, and images are taken from "./camera_cal/". This mapping is passed to `cv2.calibrateCamera()` which returns the camera matrix and distortion coefficients (amongst other things).

Applying this distortion correction to a chessboard image using the `cv2.undistort()` (in `cal_undistort` in the first code cell of the ipynb notebook) function and yields the following result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected Test Image.

Here's an example of distortion correction applied to a test image:

![alt text][image2]

#### 2. Thresholded Binary Image + Region Masking

After distortion correction, I applied a combination of color thresholds to generate a binary image. See code cell 2 of the `pipeline` function in the ipython notebook. Additionally a region mask is applied to the output of the binary image to focus subsequent parts of the pipeline to search for lines on the road in front of the car. The region mask uses the `region_of_interest` function from the Lane Finding project and vertices are hardcoded as:

```
imshape = img.shape
outer_1 = (0, imshape[0])
outer_2 = (615, 400)
outer_3 = (675, 400)
outer_4 = (imshape[1], imshape[0])

inner_1 = (350, imshape[0])
inner_2 = (650,475)
inner_3 = (700,475)
inner_4 = (1150, imshape[0])

vertices = np.array([[outer_1, outer_2, outer_3, outer_4,
         inner_4, inner_3, inner_2, inner_1]], dtype=np.int32)

masked_image = region_of_interest(combined_binary,vertices)
```
Here is an example output after applying binary masking + region thresholding:

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `warp()`, which appears in code cell 9 of the IPython notebook.  `warp` function takes as inputs an image (`img`), and returns an transformed image. Just before the function, the source (`src`) and destination (`dst`) points are hardcoded and the `M` and `Minv` transform matrices are calculated once:

```
src = np.float32([
    [200,700],
    [600,450],
    [700,450],
    [1100,700]
])
dst = np.float32([
        [375,700],
        [375,0],
        [875,0],
        [875,700]
    ])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```

The warp function simply takes an `img` as a parameter and executes a perspective transform using the M defined above.

Here's an example of the output:

![alt text][image4]

#### 4. Finding Lane Lines, Curvature, and Result

Code to find lines can be found in cells 11 and 12: `find_line_with_window` uses a histogram and 9 sliding windows to identify candidate pixels indexes for the lane line. A 2nd order polynomial is fit to the pixel positions of these indices and used to generate the points. `find_line_without_window` uses previous good fits for left and right lane indices to do a targeted search for lane lines. Curvature is calculated in code cell 13 of the ipython notebook. And, offset from the center is calculated in code cell 14. The result is the lane area shaded on the original road using the `draw_poly` function in cell 15. Here's an image with each of these steps shown:

![alt text][image5]

###Pipeline (video)

####1. Final Project Video

Here's a [link to my video result](./project_video.mp4)

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most difficulties arose from color changes of the lanes and road, as well as shadows falling over lane lines. Curved roads were also a challenge, and I don't think this pipeline performs particularly well on the challenge videos. 

Major improvements came after applying multiple masks to HSV and HLS transforms. Additionally, adding sanity checks to check for parallel lines, and small changes in radius of curvatures were also very helpful. Additionally, using previous "good fits" to do targeted searches provided a speedup to the pipeline.

To make it more robust, histogram equilization could be used to deal with contrast. Additionally, more time could be spent refining the window method for lane detection and potentially using some machine learning to predict where candidate pixels might be, and/or what combination of thresholding techniques would be good given an image of a road - That is, if we can identify shadows, discoloration and other artifacts ahead, we could potentially predict which filters to apply to the undistorted image of the road.

