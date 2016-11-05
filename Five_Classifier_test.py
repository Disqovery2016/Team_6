
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

#Feature set - Data: [BOD, DO, Conductivity]

X = [[1,8.1,89], [20.2,0.8,899], [1,8.5,122], [13.1,0.9,714], [1,9.0,195], [1,9.2,207], [14.5,4.1,1162], [1.3,9.5,236], [8.7,6.9,1211], [1.4,8.8,336], [8.9,6.4,1132],[2.1,7.9,380],[9.6,9.3,1252], [1.8,8.3,408], [17.7,5.0,1174],[9.7,9.5,1044]]

#Label set - Safe/Unsafe

Y = ['safe', 'unsafe', 'safe', 'unsafe', 'safe', 'safe', 'unsafe', 'safe', 'unsafe', 'safe', 'unsafe','safe','unsafe','safe','unsafe','unsafe']


# Calling decision tree classifier and fitting
clf1 =tree.DecisionTreeClassifier()
clfDT =clf1.fit(X, Y) 

#Calling support vector machine and fitting
clf2 = svm.SVC(probability=True)
clfSVC =clf2.fit(X, Y)  

#Calling KNeighbors classifier and fitting 
clf3 = KNeighborsClassifier(n_neighbors=3)
clfKN =clf3.fit(X, Y)  

#Calling gaussian_process classifier and fitting 
clf4 = GaussianProcessClassifier()
clfGP = clf4.fit(X, Y)

##Calling MLPClassifier and fitting 
clf5 = MLPClassifier(learning_rate='constant', learning_rate_init=0.001,)
clfMLP = clf5.fit(X, Y)

test = [3, 4, 42] #put the value you want to test NEW DATA FEEDING
#Storing results 
predictionDT = clfDT.predict (test) 
predictionSVC = clfSVC.predict (test) 
predictionKN = clfKN.predict (test) 
predictionGP = clfGP.predict (test) 
predictionMLP = clfMLP.predict (test) 

#Storing probabilities
probaDT = clfDT.predict_proba (test)
probaSVC = clfSVC.predict_proba (test) 
probaKN = clfKN.predict_proba (test) 
probaGP = clfGP.predict_proba (test) 
probaMLP = clfMLP.predict_proba (test)  


print "DT Classifier test data {} is predicted as {} with probability of {}".format(test, predictionDT, probaDT)

print "SV Classifier test data {} is predicted as {} with probability of {}".format(test, predictionSVC, probaSVC)

print "KN classifier test data {} is predicted as {} with probability of {}".format(test, predictionKN, probaKN)

print "GP classifier test data {} is predicted as {} with probability of {}".format(test, predictionGP, probaGP )

print "MLPClassifier test data {} is predicted as {} with probability of {}".format(test, predictionDT, probaDT)

