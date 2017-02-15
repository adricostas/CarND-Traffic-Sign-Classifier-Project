
# Traffic Sign Recognition


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/OccurrenciesTrainingSet.png "Occurrencies"
[image2]: ./examples/coords.png "Bounding box"
[image3]: ./examples/original.png "Original image"
[image4]: ./examples/normalized.png "Normalized image"
[image5]: ./examples/OccurrenciesAfterAugment.png "Occurrencies after augment"
[image6]: ./examples/original20.png "Image without rotation"
[image7]: ./examples/rotate20.png "Rotated image"
[image8]: ./examples/ConfusionMatrix.png "Confusion matrix"
[image9]: ./examples/downloadedImages.png "Downloaded images"
[image10]: ./examples/SoftmaxProbabilities.png "Softmax probabilities"
[image11]: ./examples/classes.png "List of images"

## Rubric Points

The rubric points of the project are shown [here](https://review.udacity.com/#!/rubrics/481/view).You can check if my code verify all of them [here](https://github.com/adricostas/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### 1. Data Set Summary & Exploration

The code for this step is contained in the section _Step 1: Dataset Summary & Exploration_ of the IPython notebook.  

I used python and numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 1.2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

In the second cell of this first section, we define two functions that will help us to visualize the images contained in the dataset (visualize) and how many they are for each class (plot_number_of_occurrencies). Below, you can see all the different classes of sign contained in the dataset.

![alt text][image11]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed according to the sign classes, that is, each bar stands for a class and its height represents the number of images of this class. As you can see, the dataset is not uniformly distributed. This could give rise to a bias in the predictions of our CNN.

![alt text][image1]

 

### 2. Design and Test a Model Architecture

#### 2. 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

In the second cell of this section, we define the necessary functions to pre-process the data. In this case, we only pre-process the images normalizing them (normalize(X)). The normalization function chosen was that present in this [lab](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/lab.ipynb)

It should be noted that I decided not to convert the images to grayscale. This decision was made because the meaning of a traffic sign is pretty related with its color. Moreover I made some trials with grayscale images and the validation accuracy was a bit lower than that achieved considering the 3 channels of color. 

In addition, I have tested the performance trimming the image with the help of the [bounding box coordinates](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) provided in the data set (coord_train = train['coords']). However, I decided to discard this technique because in Step 3, images not belonging to the German Traffic Sign Dataset don't have this coordinates available. Then, I won't be able to run the same preprocessing with them.

![alt text][image2]

Here is an example of a traffic sign image before and after normalizing.

 Original image            |  Normalized image
:-------------------------:|:-------------------------:
![alt text][image3]        |  ![alt text][image4]


#### 2.2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Currently, in this project, the training, validation and test set are given, we don't need to split the training set to achieve a validation set. 

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.

As explained in the first section, the inequality in the number of occurrencies of each class in the training set can give rise to a bad performance. That's why I decided to augment the data to make uniform this distribution. The code to augment the data can be seen in the second cell of the step 2 section. In that, we calculate the maximum number of occurrencies for one class in the data set and then we augment the number of occurrencies of each class up to the maximum, rotating the original images of this class randomly. You can see below the distribution achieved:

The difference between the original data set and the augmented data set is the following:
![alt text][image1] 

![alt text][image5]

My final training set had 86430 number of images.


Here is an example of an original image and an augmented image:

 Original image            |  Normalized and rotated image
:-------------------------:|:-------------------------:
![alt text][image6]        |  ![alt text][image7]





#### 2.3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the second cell of the _Model Architecture section_. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x64    				|
| Flatten       		| Output = 1600									|
| Fully connected		| Output = 120									|
| RELU                  |                                               |
| Fully connected		| Output = 84									|
| RELU                  |                                               |
| Dropout       		| 0.5       									|
| Fully connected		| Output = 43									|



#### 2.4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the second, third, fourth and fifth cell of the _Train, validate and test model_ section. 

To train the model, I used:

* batch size = 128
* learning rate = 0.001
* epochs = 100
* AdamOptimizer (Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods)

As you can see, I use the same parameters than in the [LeNet-Lab project](https://github.com/adricostas/CarND-LeNet-Lab/blob/master/LeNet-Lab.ipynb) except for the number of epochs. In this case, this number is increased because there are much more params (weights and biases) to tune, so the CNN need more time to reach an acceptable accuracy.

#### 2.5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the test accuracy of the model is located in the fifith cell of the _Train, validate and test model_ section.

As far as the structure of the CNN is concern, first, I started with the LeNet model developed in the [LeNet-Lab project](https://github.com/adricostas/CarND-LeNet-Lab/blob/master/LeNet-Lab.ipynb) and images in grayscale. Then, I decided to work with colored images and thus I had to change the input shape for the first layer. Afterwards, I increased the number of filters in each convolutional layer so that the CNN were able to pick up different qualities of an image. Once I had got an acceptable validation accuracy, I applied dropout after the second fully connected layer in order to avoid the overfitting and, that way, achieve a test accuracy pretty similar to the validation accuracy.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.973 
* test set accuracy of 0.960

In order to make a deeper study of the performace of the CNN, I decided to represent the confusion matrix for the test set (seventh cell of the same section). As you can see, the majority of errors are located on the top left. These are all the speed limits. It seems like the model can detect it is a speed limit but has trouble identifying the difference between the values in the sign. There are some errors in the middle zone too. These are warning signs with triangular shape related with pedestrians and animals.

![alt text][image8]

### 3. Test a Model on New Images

#### 3.1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9]

As you can see, the first image might be difficult to classify because of the white box enclosing the limit sign and the letters at the bottom. The rest of the images should be pretty easy to classiffy for the CNN. However, we know from the confusion matrix that the CNN has problems to differentiate between the speed limits sign, so the second and the third images could be difficult to classify as well.

#### 3.2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the fourth cell of the section 3.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  | Speed limit (80km/h)							| 
| Speed limit (80km/h)	| Speed limit (80km/h)							|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| General caution  		| General caution   			 				|
| Slippery Road			| No entry          							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares unfavorably to the accuracy on the test set of 0.96 but this set of new images is not comprehensive enough to draw conclusion.

#### 3.3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 6th and 7th cells (visualizations) of this section.

For all images, the model is sure about its prediction giving a probability of 1 to the first softmax probability. Here are the numbers and the visualizations.

**Speed limit 20km/h**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00 		| Speed limit (80km/h)							| 
| 9.86375781e-09		| Speed limit (120km/h)	 						|
| 2.02118722e-09		| Speed limit (60km/h)							|
| 2.12283160e-14		| Speed limit (30km/h)			 				|
| 1.81136115e-14	    | Stop              							|

The softmax probability for the actual sign is 3.78206789e-26 being that the eighth possibility considered by the CNN.

**Speed limit 80km/h**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00 		| Speed limit (80km/h)							| 
| 5.17879925e-15		| Speed limit (60km/h)	 						|
| 5.32493416e-44		| Speed limit (50km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)			 				|
| 0.00000000e+00	    | Speed limit (20km/h) 							|

**Speed limit 30km/h**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00 		| Speed limit (30km/h)							| 
| 0.00000000e+00		| Speed limit (70km/h)	 						|
| 0.00000000e+00		| Speed limit (60km/h)							|
| 0.00000000e+00		| Speed limit (50km/h)			 				|
| 0.00000000e+00	    | Speed limit (20km/h) 							|


**General caution**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00 		| General caution   							| 
| 0.00000000e+00		| Speed limit (60km/h)	 						|
| 0.00000000e+00		| Speed limit (50km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)			 				|
| 0.00000000e+00	    | Speed limit (20km/h) 							|

**No entry**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00 		| No entry          							| 
| 0.00000000e+00		| Speed limit (60km/h)	 						|
| 0.00000000e+00		| Speed limit (50km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)			 				|
| 0.00000000e+00	    | Speed limit (20km/h) 							|

As an example of visualization of softmax probabilities for an image:

![alt text][image10]


### 4. Conclusion

The accuracy achieved with the developed CNN is quite good and the results indicate that there is not overfitting. However, they could possibly be even better with another model structure using inception layers for example. However, I think that the best improvement would come with a deeper preprocessing of the images. As said in section 1, the german traffic signs dataset includes bounding box coordinates. If we trimmed the images by this bounding box the accuracy would be increased. The problem is that images not coming from that dataset do not have this coordinates available, so that, we would have to develop a method to find them and then trim the images.




```python


```


<style>
  table {margin-left: 0;}
</style>



```python

```
