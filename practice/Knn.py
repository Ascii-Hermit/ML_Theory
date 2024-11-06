import numpy as np
from collections import Counter
# class KNN:
#     def __init__(self,k=3):        
#         self.k = k

#     def fit(self,x,y):
#         self.x_train = x
#         self.y_train = y

#     def predict(self,X):
#         return [self._predict(x) for x in X ]
    
#     def _predict(self,x):
#         distances = [self.euclidian_distance(x,x_train) for x_train in self.x_train]

#         k_ind = np.argsort(distances)[:self.k]
#         k_labels = [self.y_train[i] for i in k_ind]

#         counter = Counter(k_labels)
#         return counter.most_common()[0][0]

#     def euclidian_distance(self,x,x_train):
#         return np.sqrt(np.sum(x-x_train)**2)


# class KNN:
#     def __init__(self,k=3):
#         self.k = k

#     def fit(self,x,y):
#         self.x_train = x
#         self.y_train = y

#     def predict(self,x):
#         return [self._predict(ele) for ele in x]
    
#     def _predict(self,x):
#         distances = [self.euclidian_dist(x,x_train) for x_train in self.x_train]
#         k_ind = np.argsort(distances)[:self.k]

#         k_labels = [self.y_train[i] for i in k_ind]

#         counter = Counter(k_labels)
#         return counter.most_common()[0][0]

#     def euclidian_dist(self,x,x_train):
#         return np.sqrt(np.sum(x-x_train)**2)

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure()
# plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()


# clf = KNN(k=5)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# acc = np.sum(predictions==y_test)/len(predictions)
# print(acc)

import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,x,y):
        self.x_train=x
        self.y_train=y

    def predict(self,x):
        return [self._predict(ele) for ele in x]
    
    def _predict(self,x):
        distances = [self.euclidean_dist(x,x_train) for x_train in self.x_train]
        k_ind = np.argsort(distances)[:self.k]

        k_label = [self.y_train[i] for i in k_ind]
        counter = Counter(k_label)
        return counter.most_common()[0][0]
    def euclidean_dist(self,x,y):
        return np.sqrt(np.sum((x-y)**2))
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions==y_test)/len(predictions)
print(acc)