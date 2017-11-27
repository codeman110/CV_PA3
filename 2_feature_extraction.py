from os import listdir
from os.path import join
from random import sample
import numpy as np
import imageio
import skvideo.io as skio
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import r2_score

# Data path
path_data = 'video'

# Create a list of videos
list_data = [i for i in sorted(listdir(path_data))]

# Get min width, height and number of frames
#nb_frames = []
#width = []
#height = []
#for i in list_data:
#    reader = imageio.get_reader(join(path_data,i))
#    nb_frames.append(reader.get_meta_data()['nframes'])
#    w, h = reader.get_meta_data()['size']
#    width.append(w)
#    height.append(h)
#norm_nb_frames = min(nb_frames)
norm_width = 120 #min(width)
norm_height = 90 #min(height)

temp = []
for i in list_data:
    temp.append(i[:][:3])
    
temp = np.unique(np.array(temp))
list_train = []
list_val = []
list_test = []
for i in temp:
    new_list = []
    for j in list_data:
        if j[0:3] == i:
            new_list.append(j)
    # Randomly select two videos from each class into validation and test list
    rnd = sample(new_list,2)
    # Add to validation list
    list_val.append(rnd[0])
    # Add to test list
    list_test.append(rnd[1])
    # Pop the elements
    new_list.pop(new_list.index(rnd[0]))
    new_list.pop(new_list.index(rnd[1]))
    list_train.append(new_list)
    
# Flatten the train list
list_train_flat = [item for sublist in list_train for item in sublist]

# Take out labels for train set
label_train = []
for i in list_train_flat:
    vid = skio.vread(join(path_data,i), as_grey=True, outputdict={'-sws_flags': 'bilinear', '-s': str(norm_width)+'x'+str(norm_height)})
    fr, ht, wd, ch = vid.shape
    # Reshape the data
    a = []
    for j in range(fr):
        image = vid[j].reshape(ht, wd, ch)
        a.append([image, ht, wd, ch, int(i[0:3])])
    label_train.append(a)
    
# Take out labels for validation set
label_val = []
for i in list_val:
    vid = skio.vread(join(path_data,i), as_grey=True, outputdict={'-sws_flags': 'bilinear', '-s': str(norm_width)+'x'+str(norm_height)})
    fr, ht, wd, ch = vid.shape
    # Reshape the data
    a = []
    for j in range(fr):
        image = vid[j].reshape(ht, wd, ch)
        a.append([image, ht, wd, ch, int(i[0:3])])
    label_val.append(a)

# Take out labels for test set
label_test = []
for i in list_test:
    vid = skio.vread(join(path_data,i), as_grey=True, outputdict={'-sws_flags': 'bilinear', '-s': str(norm_width)+'x'+str(norm_height)})
    fr, ht, wd, ch = vid.shape
    # Reshape the data
    a = []
    for j in range(fr):
        image = vid[j].reshape(ht, wd, ch)
        a.append([image, ht, wd, ch, int(i[0:3])])
    label_test.append(a)

# HOG features for training set
hog_train = []
count = 0
for i in label_train:
    count += 1
    print count ########################
    for j,_ in i[:][:][0]:
        if count == 1:
            print j
        image = j.reshape(norm_height,norm_width)
        feats,_ = hog(image, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(1,1), block_norm='L2', visualise=True)
    hog_train.append(feats.tolist())

# HOG features for validation set
hog_val = []
for i in label_val:
    count += 1
    print count ########################
    for j in i[:][:][0]:
        image = j.reshape(norm_height,norm_width)
        feats,_ = hog(image, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(1,1), block_norm='L2', visualise=True)
    hog_val.append(feats.tolist())

# HOG features for test set   
hog_test = []
for i in label_test:
    count += 1
    print count ########################
    for j in i[:][:][0]:
        image = j.reshape(norm_height,norm_width)
        feats,_ = hog(image, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(1,1), block_norm='L2', visualise=True)
    hog_test.append(feats.tolist())
    
'''    
# SVM
# Data
data_train = np.array(hog_train)
data_val = np.array(hog_val)
data_test = np.array(hog_test)
# Labels
lbl_train = np.array([i[:][4] for i in label_train]).ravel()
lbl_val = np.array([i[:][4] for i in label_val]).ravel()
lbl_test = np.array([i[:][4] for i in label_test]).ravel()

# Linear kernel
# Training the model
svm_lin = svm.SVC(kernel='linear', C=1.0, cache_size=4096)
fit_lin = svm_lin.fit(data_train,lbl_train)

# Validate the model
pred_val = svm_lin.predict(data_val)
acc_val = r2_score(lbl_val,pred_val)*100
print 'Validation accuracy -> ' + str(acc_val)

# Testing the model
pred_test = svm_lin.predict(data_test)
acc_test = r2_score(lbl_test,pred_test)*100
print 'Test accuracy -> ' + str(acc_test)

# Sensitivity and specificity
t, f = 0,0
for i in range(len(lbl_test)):
    if lbl_test[i] == pred_test[i]:
        t += 1
    else:
        f += 1
sens = float(t/len(lbl_test))
spec = float(f/len(lbl_test))
print 'Test sensitivity -> ' + str(sens)
print 'Test specificity -> ' + str(spec)
'''