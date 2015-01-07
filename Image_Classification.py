import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

#Load the images
all_instance_filenames = []
all_instance _targets = []
for f in glob.glob('train/*.jpg'):
	target = 1 if 'cat' in f else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)
#Convert the images to grayscale, and extract the SURF descriptors
surf_features = []
counter = 0
for f in all_instance_filenames:
	print 'Reading image:', f
	image = mh.imread(f, as_grey=True)
	surf_features.append(surf.surf(image)[:,5:])
train_len = int(len(all_instance_filenames)*.60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]