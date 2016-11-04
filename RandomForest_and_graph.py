import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
#Feature set - Data: [BOD, DO, Conductivity]

X = [[1,8.1,89], [20.2,0.8,899], [1,8.5,122], [13.1,0.9,714], [1,9.0,195], [1,9.2,207], [14.5,4.1,1162], [1.3,9.5,236], [8.7,6.9,1211], [1.4,8.8,336], [8.9,6.4,1132],[2.1,7.9,380],[9.6,9.3,1252], [1.8,8.3,408], [17.7,5.0,1174],[9.7,9.5,1044]]

#Label set - Safe/Unsafe
Y = ['safe', 'unsafe', 'safe', 'unsafe', 'safe', 'safe', 'unsafe', 'safe', 'unsafe', 'safe', 'unsafe','safe','unsafe','safe','unsafe','unsafe']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

clf = RandomForestClassifier(n_estimators=1)

y_pred = clf.fit(X_train, y_train).predict(X_test)
print y_pred

print  clf.score(X_test, y_test)
print  confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)


plt.subplot(321)
plt.title("One informative feature, one cluster per class", fontsize='small')
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

plt.subplot(322)
plt.title("Multi-class, two informative features, one cluster",
          fontsize='small')
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)



plt.show()

