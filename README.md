# INCOMPLETE
# CV - PA3
Steps:
1. Download the UCF Sports Action Data Set from the following link - http://crcv.ucf.edu/data/ucf_sports_actions.zip
2. Extract the zip file in the same folder as the python files.
3. Run the 1_preprocessing.py to copy the videos into a new folder followed by 2_feature_extraction.py.

Notes:
1. Run the 1_preprocessing.py only once. This file will copy all the videos from the original dataset directory to new directory. It will also create videos from images where videos are missing. It will rename the videos in the new directory in the format  xxx_yyy.avi where xxx is the class number (ranging from 000-012) and yyy is the video number in that partcular class. This format 	is used to avoid cration of a separate csv file where information of video and their labels are kept.
	
2. We are considering 13 classes as given in the directory of UCF Sports Action Data Set instead of 10 classes.

3. From each of the 13 action classes a video is chosen randomly to be used as test data.

4. Histogram of Gradients is used to extract features from the images. To decrease the computation time, videos are resized to 90x120. The parameters of HoG are 
  orientations=9,
  pixels_per_cell=(8,8),
  cells_per_block=(1,1),
  block_norm='L2'


5. 
