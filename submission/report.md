# Traffic Sign Recognition

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I plot 25 randomly sampled images from the training dataset with corresponding labels (in the format of *label_id-sign_name*).

![alt text](./train_sample_images.png)

Below is a bar chart showing the distribution of the training data. We can tell that some classes have far fewer training images than other classes.

![alt text](./train_set_statistics.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

- As a first step, I decided to convert the images to grayscale because the color information on the traffic signs seems not as important as the shape of the signs.
- Then, I did a histogram equalization for each image using openCV function `cv2.equalizeHist(image)` to increase the contrast because it seems a lot of the images were taken with very low exposure.
- Next, I applied a Gaussian blur with kernel (3, 3) to each image to reduce noise using `cv2.GaussianBlur(image, (3, 3), 0)`. The resulting sample images (same as above) are shown below

![alt text](./train_sample_images_processed.png)

- Then, I converted the image into floating point representation.
- Finally, I applied normalization to every image in the dataset such that each image has zero mean and unit variance. The `adjusted_std` variable is borrowed from tensorflow implementation of the `tf.image.per_image_standardization` to avoid the case of uniform images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer | Description |
|:---------------------:|:---------------------------------------------:|
| Input | 32x32x3 RGB image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU | |
| Max pooling | 2x2 stride, outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU | |
| Max pooling | 2x2 stride, outputs 5x5x16 |
| Fully connected | inputs flattened 5x5x16=400, outputs = 120|
| RELU | |
| Dropout | keep_prob = 0.5 for training, 1.0 for testing |
| Fully connected | outputs = 84 |
| RELU | |
| Dropout | keep_prob = 0.5 for training, 1.0 for testing |
| Fully connected | outputs = 43 |
| RELU | |
| Softmax | |

The network architecture is very similar to the LeNet except the final output layer size is changed to 43. Another notifiable difference is the inclusion of two dropout layers. This greatly reduced overfitting.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used optimizer that implements [Adam algorithm](http://arxiv.org/abs/1412.6980) with a learning rate 0.001. The batch size is chosen to be 256. I ran the optimization for 200 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ~0.999
* validation set accuracy of 0.971
* test set accuracy of 0.946

<!-- If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? -->

If a well known architecture was chosen:
* What architecture was chosen?

I started with original LeNet, just modifying the output layer to have size 43 instead of 10. The result is not bad but the it is clearly overfitting. So I added dropout layers after the fully connected layer with relu activation. This greatly reduced the overfitting.

* Why did you believe it would be relevant to the traffic sign application?

This traffic sign dataset is very similar to the minist dataset, in that they have very similar image size. The training set is actually smaller that that of minist dataset (34799 vs 60000) but have more classes (43 vs 10). This means the size of a LeNet is enough for this problem with a potential overfitting problem. No deep architecture like ResNet is needed.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The training accuracy is almost 1.0 and the validation accuracy stays above 0.97 after 100 epochs. The testing accuracy is 0.946 which is as expected given the training and validation accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I randomly chosen from the test set which are not seen during training process.

![alt text](test_sample_images.png)

The last image may be difficult to classify because the image contrast is very low and the image is taken at an angle that make the sign not round.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following image shows the top 5 softmax probabilities of the 5 test images.

![alt text](top5_prediction_on_test_images.png)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The following image shows the feature maps of the two convolution layers of each image:

![alt text](visualization_of_feature_maps.png)
