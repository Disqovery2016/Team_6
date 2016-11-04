import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#Feature set - Data: [BOD, DO, Conductivity]

X = [[1,25,100],[1,5,900],[1,20,100],[1,24,100],[1,6,1000],[1,7,1100],[1,9,1300], [1,10,1400], [1,11,1500],[1,13,1700], 
[1,14,1800],[1,15,1900],[1,16,2000],[1,17,2100],[1,1,100],[1,2,200],[1,3,300],[1,4,400],[1,5,500],[1,6,600],[1,7,700],[1,8,800],[1,9,900],[1,8.1,89], [20.2,0.8,899], [1,8.5,122], [13.1,0.9,714], [1,9.0,195], [1,9.2,207], [14.5,4.1,1162], [1.3,9.5,236], [8.7,6.9,1211], [1.4,8.8,336], [8.9,6.4,1132],[2.1,7.9,380],[9.6,9.3,1252], [1.8,8.3,408], [17.7,5.0,1174],[9.7,9.5,1044]]

#Label set - Safe/Unsafe
Y = ['unsafe','safe','unsafe','safe','unsafe','safe','safe','safe','safe','safe','safe','safe','safe','safe','unsafe','unsafe','unsafe','unsafe','unsafe','unsafe','unsafe','unsafe','unsafe','safe', 'unsafe', 'safe', 'unsafe', 'safe', 'safe', 'unsafe', 'safe', 'unsafe', 'safe', 'unsafe','safe','unsafe','safe','unsafe','unsafe']


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

print "training data"
print X_train
print "X test deta"
print X_test
print "Y training data"
print y_train
print "Y testing data"
print y_test

clf = RandomForestClassifier(n_estimators=1)

y_pred = clf.fit(X_train, y_train).predict(X_test)
print y_pred

print  clf.score(X_test, y_test)
print  confusion_matrix(y_test, y_pred)


