import scipy.io
import os
import glob, os

annotation_dir = '../ObjectNet3D/Annotations'
files = glob.glob(annotation_dir + '/*.mat')

classes = set([])

#for f in files:

f = files[15]
mat = scipy.io.loadmat(f)
record = mat['record']

# Get the filename
fname = record['filename'][0, 0][0]

# Get the image size
im_size = record['imgsize'][0, 0][0]

# Get all the objects
objects = mat['record']['objects'][0, 0]
num_objects = objects.shape[1]
for i in range(num_objects):
    class_name = objects[0, i]['class'][0]
    bbox = objects[0, i]['bbox'][0]

    # Get the viewpoint information
    viewpoint = objects[0, i]['viewpoint'][0, 0]
    azimuth_coarse = viewpoint['azimuth_coarse'][0, 0]
    elevation_coarse = viewpoint['elevation_coarse'][0, 0]
    theta = viewpoint['theta'][0, 0]

    print(objects[0, i]['difficult'][0, 0])
    print(objects[0, i]['occluded'][0, 0])
    print(objects[0, i]['truncated'][0, 0])