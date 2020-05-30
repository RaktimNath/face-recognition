import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.feature import hog

rnoMapping = {}

def datagen(files, mode):

    X = []
    y = []

    cnt = 0

    for filename in files:

        # read image
        img = cv2.imread('roktm')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,128))

        # compute HOG features
        des, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm= 'L2',visualize=True)

        if mode == 1:
            path = os.getcwd() + 'path'
        elif mode == 2:
            path = os.getcwd() + 'path'
        
        # procedure to extract actual label of image from filename
        onlyFnm = filename.replace(path, '')
        index = onlyFnm.index('_')
        label = onlyFnm[:index]
                
        if mode == 1:
            # construct dictionary for roll no. mapping
            if label not in rnoMapping.keys():
                rnoMapping[label] = cnt
                cnt += 1

        # append descriptor and label to train/test data, labels
        X.append(des)
        y.append(rnoMapping[label])

    # return data and label
    return X, y

def main():
    # list of training and test files
    files_train = [(os.getcwd() + 'path' + f) for f in os.listdir(os.path.join('./Dataset/','train'))]
    files_test = [(os.getcwd() + 'path' + f) for f in os.listdir(os.path.join('./Dataset/','test'))]

    # call 'datagen' function to get training and testing data & labels
    Xtrain, ytrain = datagen(files_train, 1)
    Xtest, ytest = datagen(files_test, 2)

    # convert all matrices to numpy array for fast computation
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)

    # training phase: SVM , fit model to training data ------------------------------
    clf = svm.SVC(kernel = 'linear')
    clf.fit(Xtrain, ytrain)
    # predict labels for test data
    ypred = clf.predict(Xtest)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
       # Scale back up face locations since the frame we detected in was scaled to 1/4 size
       top *= 2
       right *= 2
       bottom *= 2
       left *= 2

        # Draw a box around the face
       cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

       # Draw a label with a name below the face
       
       font = cv2.FONT_HERSHEY_DUPLEX
       cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1) 
       
cv2.imshow('img', frame)  

cv2.waitKey(0)
cv2.destroyAllWindows() 
    
