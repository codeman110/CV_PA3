# INCOMPLETE
# COMPUTER VISION - PROGRAMMING ASSIGNMENT 3
## Action Recognition using Support Vector Machines
Steps:
1. Download the [UCF Sports Action Data Set](http://crcv.ucf.edu/data/ucf_sports_actions.zip).
2. Extract the zip file in the same folder as the python files.
3. Run the 1_preprocessing.py to copy the videos into a new folder followed by 2_feature_extraction.py.

Notes:
1. Libraries used
```python
import imageio
import numpy as np
from sklearn import svm
from shutil import copy2
from random import sample
import skvideo.io as skio
from skimage.feature import hog
from os import listdir, makedirs
from sklearn.metrics import r2_score
from os.path import join, isdir, exists
```

2. **Run the 1_preprocessing.py only once.** This file will copy all the videos from the original dataset directory to new directory. It creates videos from images and where images are not availble, it copies the video. It will rename the videos in the new directory in the format  xxx_yyy.avi where xxx is the class number (ranging from 000-012) and yyy is the video number in that partcular class. This format is used to avoid cration of a separate csv file where information of video and their labels are kept. The imageio library is used to convert images to video file.
	
3. We are considering 13 classes as given in the directory of UCF Sports Action Data Set instead of 10 classes.

4. Histogram of Gradients is used to extract features from the images. To decrease the computation time, videos are resized to 120x90 and grayscale channel is used. The parameters of HoG are
   - orientations -> Number of orientation bins.
   - pixels_per_cell -> Size (in pixels) of a cell
   - cells_per_block -> Number of cells in each block
   - block_norm -> Block normalization method
```python
feats,_ = hog(image, orientations=9, pixels_per_cell=(8,8),
              cells_per_block=(1,1), block_norm='L2', visualise=True)
```
5. The SVM classifier is used to classification. Linear kernel is used. The following parametes are used
   - kernel='linear'
   - C=1.0
   - probability=True
   - cache_size=4096
```python
# Training the model
svm_lin = svm.SVC(kernel='linear', C=1.0, probability=True, cache_size=4096)
fit_lin = svm_lin.fit(data_train,lbl_train)

# Testing the model
pred_test = svm_lin.predict(data_test)
```
6. The Leave-one-out (LOO) cross-validation scheme is used. This scenario takes out one sample video for testing and trains using all of the remaining videos of an action class. This is performed for every sample video in a cyclic manner, and the overall accuracy is obtained by averaging the accuracy of all iterations.
Here the ```python MAX_ITR``` defines the number of epochs the SVM should run. Since there are 150 videos, the SVM should run 150 times if LOO cross-validation scheme is used. The disadvantage is that it takes a lot of time (8.9 hours approx.). We ran the SVM for 10 times.
7. Specificity, sensitivity and accuracy are obtained after testing the data.
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
