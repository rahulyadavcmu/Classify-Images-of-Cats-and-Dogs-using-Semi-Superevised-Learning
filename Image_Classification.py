import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

#Load the images
all_instance_filenames = []
all_instance_targets = []
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

#Split the images into training and testing data
train_len = int(len(all_instance_filenames)*.60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]

#Group the extracted descriptors into 300 clusters. Use MiniBatchKMeans
#to compute the distances to the centroids for a sample of the instances.
n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'
estimator = MiniBatchKMeans(n_clusters=n_clusters)
estimator.fit_transform(X_train_surf_features)

#Construct feature vectors for training and testing data. Find the cluster
#associated with each of the extracted SURF descriptors, and count them using 
# np.bincount()
X_train = []
for instance in surf_features[:train_len]:
	clusters = estimator.predict(instance)
	features = np.bincount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters-len(features))))
	X_train.append(features)

X_test = []
for instance in surf_features[train_len:]:
	clusters = estimator.predict(instance)
	features = np.bincount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters-len(features))))
	X_test.append(features)

#Train a logistic regression classifier on the feature vectors and targets,
#and assess its precision, recall, and accuracy.
clf = LogisticRegression(C=0.001, penalty='l2')
clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)

















