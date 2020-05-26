# import the necessary packages

from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os
#import argparse
from localbinarypatterns import LocalBinaryPatterns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import numpy as np



trainpath = 'Datasetorganized/train'
testPath = 'Datasetorganized/test'

## construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--training", required=True,
#    help="path to the training images")
#ap.add_argument("-e", "--testing", required=True,
#    help="path to the tesitng images")
#
##insert number of neighbors for knn
#ap.add_argument("-k", "--neighbors", type=int, default=1,
#	help="# of nearest neighbors for classification")
#
#ap.add_argument("-j", "--jobs", type=int, default=-1,
#	help="# of jobs for k-NN distance (-1 uses all available cores)")
#
#args = vars(ap.parse_args())

desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
# loop over the training images
#for imagePath in paths.list_images(args["training"]):
for imagePath in paths.list_images(trainpath):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
#print(labels)
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42, dual=False)
model.fit(data, labels)

# train KNN algorithm on the data
modelknn = KNeighborsClassifier(n_neighbors=5,
	n_jobs=-1)
#modelknn = KNeighborsClassifier(n_neighbors=args["neighbors"],
#	n_jobs=args["jobs"])
modelknn.fit(data, labels)

# voting classifier
voting_clf = VotingClassifier(
estimators=[('svm', model), ('knn', modelknn)],
voting='hard'
)
voting_clf.fit(data, labels)

#for prediction
test_labels=[]
test_data=[]
# loop over the testing images
#for imagePath in paths.list_images(args["testing"]):
for imagePath in paths.list_images(testPath):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    name = os.path.basename(imagePath)
    name2 = str(os.path.splitext(name)[0])
    test_labels.append(name2.split('_')[:-1])
    test_data.append(hist)
#    predictionsvm = model.predict(test_data)
#    print("predictionsvm",predictionsvm)
#    predictionknn = modelknn.predict(test_data)
#    print("predictionknn",predictionknn)
#    predictionvote = voting_clf(test_data)

print(test_labels)
for clf in (model, modelknn, voting_clf):
    y_pred = clf.predict(test_data)
#    print(y_pred)
    print(clf.__class__.__name__, accuracy_score(test_labels, y_pred)*100)
#print(model.__class__.__name__, accuracy_score(test_labels, predictionsvm)*100)
#print(modelknn.__class__.__name__, accuracy_score(test_labels, predictionknn)*100)
