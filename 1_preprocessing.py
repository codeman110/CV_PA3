from os import listdir, makedirs
from os.path import join, isdir, exists
from shutil import copy2
import numpy as np
import imageio

# Original data path
path = 'ucf_sports_actions/ucf_action'

# Create new directory for videos
path_new = 'video'
if not exists(path_new):
    makedirs(path_new)

# List of action classes
classes = sorted(listdir(path))

# Traverse each action class
for i in classes:
    
    # Get action paths
    path_action = join(path, i)
    
    # Get current action sub-folders
    ac_data = sorted(listdir(path_action))

    for j in ac_data:
        # Create path for each action sub-folder
        path_data = join(path_action, j)

        # Create a list of files and folder of each action sub-folder
        file_all = sorted(listdir(path_data))
       
        # Create a list of JPG files
        file_jpg = [join(path_data,q) for q in file_all if (not isdir(q)) and q.endswith('.jpg')]
       
        # Assign new name to videos
        file_avi = str(classes.index(i)).zfill(3)+'_'+str(j).zfill(3)+'.avi'
        
        # Create Video from JPG files
        if (len(file_jpg) != 0):
            # Open Images in JPG list
            frames = [imageio.imread(k) for k in file_jpg]
            frames = np.array(frames)
        
            # Save videos to new destination
            writer = imageio.get_writer(join(path_new,file_avi), fps=10, macro_block_size=None)
            for _,l in enumerate(frames):
                writer.append_data(l)
            writer.close()
        else:
            for m in file_all:
                if m.endswith('.avi'):
                    file_vid = join(path_data,m)
                    copy2(file_vid,join(path_new,file_avi))
            