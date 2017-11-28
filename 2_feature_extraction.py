from os import listdir
from os.path import join
from random import shuffle
import numpy as np
import skvideo.io as skio
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import r2_score
import time

# Data path
path_data = 'video'
MAX_ITER = 10 # Number of instances to run the Leave-one-out

# Create a list of videos
list_data = [i for i in sorted(listdir(path_data))]

norm_width = 120 # Normalised width
norm_height = 90 # Normalised height

# Shuffle the dataset
shuffle(list_data)

# Make a list of labels and frames
list_labelled_frames = []
for i in list_data:
    t1 = []
    label = i[:][:3]        
    vid = skio.vread(join(path_data,i), as_grey=True, outputdict={'-sws_flags': 'bilinear', '-s': str(norm_width)+'x'+str(norm_height)})
    f,h,w,c = vid.shape
    t1.append(label)
    t2 = []
    for j in vid:
        image = j.reshape(h, w, c)
        t2.append(image)
    t1.append(t2)
    list_labelled_frames.append(t1)
    
# Extracting HOG features from frames
count = 0
list_hog = []
for i in list_labelled_frames:
    count += 1
    print count
    t1 = []
    label = i[0]
    t1.append(label)
    t2 = []
    for j in i[1]:
        image = j.reshape(norm_height,norm_width)
        feats,_ = hog(image, orientations=9, pixels_per_cell=(8,8),
                  cells_per_block=(1,1), block_norm='L2', visualise=True)
        t2.append(feats.tolist())
    t1.append(np.array(t2))
    list_hog.append(t1)
    
# SVM
total_acc = 0
total_sens = 0
total_spec = 0
for i in range(MAX_ITER):
    print '~'*100
    print 'Epoch - %d' % (i)
    # Restructure the data and labels
    data_train = []
    data_test = []
    label_train = []
    label_test = []
    for idx, val in enumerate(list_hog):
        if idx == i:
            label = val[0]
            nb_label = val[1].shape[0]
            data_test.append(np.array(val[1]))
            for j in range(nb_label):
                label_test.append(int(label))
        else:
            label = val[0]
            nb_label = val[1].shape[0]
            data_train.append(np.array(val[1]))
            for j in range(nb_label):
                label_train.append(int(label))
    data_train = np.vstack(data_train)
    data_test = np.vstack(data_test)
        
    # Linear kernel
    # Training the model
    start = time.time()
    svm_lin = svm.SVC(kernel='linear', C=1.0, probability=True, cache_size=4096)
    fit_lin = svm_lin.fit(data_train,label_train)
    stop = time.time()
    print 'Time taken to train -> %d' % (stop-start)
    
    # Testing the model
    pred_test = svm_lin.predict(data_test)
    acc_test = r2_score(label_test,pred_test)*100
    print 'Test accuracy -> ' + str(acc_test)
    
    # Sensitivity and specificity
    t, f = 0.0,0.0
    for i in range(len(label_test)):
        if (label_test[i] == pred_test[i]):
            t += 1
        else:
            f += 1
    sens = float(t/len(label_test))
    spec = float(f/len(label_test))
    print 'Test sensitivity -> ' + str(sens)
    print 'Test specificity -> ' + str(spec)
    total_acc += acc_test
    total_sens += sens
    total_spec += spec
print 'Average test accuracy -> ' + str(total_acc/MAX_ITER)
print 'Average test sensitivity -> ' + str(total_sens/MAX_ITER)
print 'Average test specificity -> ' + str(total_spec/MAX_ITER)
