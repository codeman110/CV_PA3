# COMPUTER VISION - PROGRAMMING ASSIGNMENT 3
## Action Recognition using Support Vector Machines
Steps:
1. Download the [UCF Sports Action Data Set](http://crcv.ucf.edu/data/ucf_sports_actions.zip).
2. Extract the zip file in the same folder as the python files.
3. Run the 1_preprocessing.py to copy the videos into a new folder followed by 2_feature_extraction.py.

Data structures:
1. We used ```skvideo.io``` library to extract frames from videos. A dataset of frames and their labels are created in the following manner. We created a list of list of lists to hold the data. The innermost list contains all the frames of a particular video. The outer list contains the label and the innermost list. The outermost list contains a list of previous list. So basically the outermost list represents all the 150 videos.

2. The HoG dataset is also created in the same manner. It is also a list of list of lists. Only difference is that, the innermost list contains HoG representaions of frames insted of frames.

3. Just before the HoG features are fed into the classifier, the dataset is again restructured. This time the innermost list of the HoG features of videos are put in a vertical stack. Basically it is numpy array. If the first video has 55 frames and the second one has 60, the dataset will have 55 HoG features of the first video followed by 60 features of second one and so on. The features are stacked vertically. The labels are also created in such a manner so that it will correspond to the HoG features. In this case, it will be list containing 55 labels of the first video followed by 60 of second one and so on.


Notes:
1. Libraries used
```python
import time
import imageio
import numpy as np
from sklearn import svm
from shutil import copy2
import skvideo.io as skio
from random import shuffle
from skimage.feature import hog
from os import listdir, makedirs
from sklearn.metrics import r2_score
from os.path import join, isdir, exists
```

2. **Run the 1_preprocessing.py only once.** This file will copy all the videos from the original dataset directory to new directory. It creates videos from images and where images are not availble, it copies the video. It will rename the videos in the new directory in the format  xxx_yyy.avi where xxx is the class number (ranging from 000-012) and yyy is the video number in that partcular class. This format is used to avoid creation of a separate csv file where information of video and their labels are kept. The imageio library is used to convert images to video file.
	
3. We are considering 13 classes as given in the directory of UCF Sports Action Data Set instead of 10 classes.

4. Shuffle the dataset.

5. Extract frames from videos using ```skvideo.io``` library.
```python
vid = skio.vread(join(path_data,i),
                 as_grey=True,
		 outputdict={'-sws_flags': 'bilinear', '-s': str(norm_width)+'x'+str(norm_height)})
```
6. Histogram of Gradients is used to extract features from the images. To decrease the computation time, videos are resized to 120x90 and grayscale channel is used. The parameters of HoG are
   - orientations -> Number of orientation bins.
   - pixels_per_cell -> Size (in pixels) of a cell
   - cells_per_block -> Number of cells in each block
   - block_norm -> Block normalization method
```python
feats,_ = hog(image, orientations=9, pixels_per_cell=(8,8),
              cells_per_block=(1,1), block_norm='L2', visualise=True)
```
7. The SVM classifier is used for classification. The following parametes are used
   - kernel -> Specifies the kernel type to be used in the algorithm. Here linear kernel is used. 
   - C -> Penalty parameter C of the error term.
   - probability -> Whether to enable probability estimates.
   - cache_size= -> Specify the size of the kernel cache (in MB).
```python
# Training the model
svm_lin = svm.SVC(kernel='linear', C=1.0, probability=True, cache_size=4096)
fit_lin = svm_lin.fit(data_train,lbl_train)

# Testing the model
pred_test = svm_lin.predict(data_test)
```
8. The Leave-one-out (LOO) cross-validation scheme is used. This scenario takes out one sample video for testing and trains using all of the remaining videos of an action class. This is performed for every sample video in a cyclic manner, and the overall accuracy is obtained by averaging the accuracy of all iterations.
Here the ```MAX_ITR``` defines the number of epochs the SVM should run. Since there are 150 videos, the SVM should run 150 times if LOO cross-validation scheme is used. The disadvantage is that it takes a lot of time (8.9 hours approx.). We ran the SVM for 10 epochs (35 mins).

9. Specificity, sensitivity and accuracy are obtained after testing the data.
```python
# Sensitivity, specificity and accuracy
t, f = 0.0,0.0
for i in range(len(lbl_test)):
    if (lbl_test[i] == pred_test[i]):
        t += 1
    else:
        f += 1
sens = float(t/len(lbl_test))
spec = float(f/len(lbl_test))
acc_test = r2_score(lbl_test,pred_test)*100
```
After executing the SVM for ```MAX_ITR``` times, the average accuracy, sensitivity and specificity is calculated. Increasing the ```MAX_ITR``` will also yield good accuracy.
