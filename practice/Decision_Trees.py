import numpy as np
from collections import Counter

# class Node:
#     def __init__(self, features=None, threshold=None, left=None, right=None, value=None):
#         self.features = features
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value

#     def is_leaf_node(self):
#         return self.value is not None

# class DecisionTree:
#     def __init__(self, min_samples=2, max_depth=10):
#         self.min_samples = min_samples
#         self.max_depth = max_depth
#         self.root = None

#     def fit(self, X, y):
#         self.root = self._grow_tree(X, y)

#     def _grow_tree(self, X, y, depth=0):
#         n_samples, n_feats = X.shape
#         n_labels = len(np.unique(y))

#         # Base case
#         if n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples:
#             leaf_value = self._most_common_label(y)
#             return Node(value=leaf_value)

#         # Best split
#         best_thresh, best_feature = self._find_best_split(X, y)

#         # Child nodes
#         left_ind, right_ind = self._split(X[:, best_feature], best_thresh)
#         left = self._grow_tree(X[left_ind], y[left_ind], depth + 1)
#         right = self._grow_tree(X[right_ind], y[right_ind], depth + 1)
#         return Node(features=best_feature, threshold=best_thresh, left=left, right=right)

#     def _find_best_split(self, X, y):
#         n_feats = X.shape[1]
#         best_gain = -1
#         split_thresh, split_idx = None, None

#         for idx in range(n_feats):
#             x_column = X[:, idx]
#             thresholds = np.unique(x_column)

#             for thr in thresholds:
#                 gain = self._information_gain(y, x_column, thr)

#                 if gain > best_gain:
#                     best_gain = gain
#                     split_thresh = thr
#                     split_idx = idx

#         return split_thresh, split_idx

#     def _information_gain(self, y, x_column, threshold):
#         # Parent entropy
#         parent_entropy = self._entropy(y)

#         # Split children
#         left_ind, right_ind = self._split(x_column, threshold)
#         if len(left_ind) == 0 or len(right_ind) == 0:
#             return 0

#         # Children entropy
#         n = len(y)
#         n_l, n_r = len(left_ind), len(right_ind)
#         e_l, e_r = self._entropy(y[left_ind]), self._entropy(y[right_ind])
#         child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

#         return parent_entropy - child_entropy

#     def _split(self, x_column, threshold):
#         left_ind = np.argwhere(x_column <= threshold).flatten()
#         right_ind = np.argwhere(x_column > threshold).flatten()
#         return left_ind, right_ind

#     def _entropy(self, y):
#         hist = np.bincount(y)
#         hist = hist / len(y)
#         return -np.sum([p * np.log(p) for p in hist if p > 0])

#     def _most_common_label(self, y):
#         counter = Counter(y)
#         value = counter.most_common(1)[0][0]
#         return value

#     def predict(self, X):
#         return np.array([self._traverse_tree(x, self.root) for x in X])
    
#     def _traverse_tree(self, x, node):
#         if node.is_leaf_node():
#             return node.value
        
#         if x[node.features] < node.threshold:
#             return self._traverse_tree(x, node.left)
#         return self._traverse_tree(x, node.right)

# # Example usage
# from sklearn import datasets
# from sklearn.model_selection import train_test_split

# data = datasets.load_breast_cancer()
# X, y = data.data, data.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234
# )

# clf = DecisionTree(max_depth=10)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# def accuracy(y_test, y_pred):
#     return np.sum(y_test == y_pred) / len(y_test)

# acc = accuracy(y_test, predictions)
# print("Accuracy:", acc)

# class Node:
#     def __init__(self,features=None,threshold=None,left=None,right=None,value=None):
#         self.features = features
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value

#     def is_leaf_node(self):
#         return self.value is not None
    
# class DT:
#     def __init__(self,max_depth = 10, min_samples = 2):
#         self.max_depth = max_depth
#         self.min_samples = min_samples
#         self.root = None

#     def fit(self,x,y):
#         self.root = self.grow_tree(x,y)

#     def grow_tree(self,x,y,depth=0):        
#         n_samples, n_features = x.shape
#         n_labels = len(np.unique(y))

#         if n_labels == 1 or n_samples<self.min_samples or depth>=self.max_depth:
#             leaf_value = self._most_common_label(y)
#             return Node(value=leaf_value)
        
#         best_thresh,best_feature = self._best_feature_split(x,y)

#         left_ind,right_ind = self._split(x[:,best_feature],best_thresh)
#         left = self.grow_tree(x[left_ind],y[left_ind],depth+1)
#         right = self.grow_tree(x[right_ind],y[right_ind],depth+1)
#         return Node(features=best_feature,threshold=best_thresh,left=left,right=right)

#     def _best_feature_split(self,x,y):
#         best_gain = -1
#         split_thresh,split_ind = -1,-1
#         n_features = x.shape[1]

#         for feat_ind in range(n_features):
#             x_col = x[:,feat_ind]
#             threshold = np.unique(x_col)

#             for thr in threshold:
#                 gain = self._gain_ratio(x_col,y,thr)

#                 if gain>best_gain:
#                     best_gain = gain
#                     split_ind = feat_ind
#                     split_thresh = thr

#         return split_thresh, split_ind     

#     def _gini_impurity(self, y):
#         labels, counts = np.unique(y, return_counts=True)
#         probabilities = counts / counts.sum()
#         return 1 - np.sum(probabilities ** 2)      
    
#     def _gain_ratio(self, x_col, y, threshold):
#         # Calculate information gain
#         information_gain = self._information_gain(x_col, y, threshold)
        
#         # Calculate split information
#         left_indices, right_indices = self._split(x_col, threshold)
#         num_samples = len(y)
        
#         # Calculate the proportions of the splits
#         left_ratio = len(left_indices) / num_samples
#         right_ratio = len(right_indices) / num_samples
        
#         # Calculate split information
#         split_info = 0
#         for ratio in [left_ratio, right_ratio]:
#             if ratio > 0:
#                 split_info -= ratio * np.log2(ratio)  # Use log base 2 for consistency
        
#         # Prevent division by zero
#         if split_info == 0:
#             return 0
        
#         # Return gain ratio
#         return information_gain / split_info
    
#     def _gini_index(self, x_col, y, threshold):
#         left_ind, right_ind = self._split(x_col, threshold)
#         n = len(y)
#         n_l, n_r = len(left_ind), len(right_ind)

#         # Calculate Gini for left and right splits
#         gini_l = self._gini_impurity(y[left_ind]) if n_l > 0 else 0
#         gini_r = self._gini_impurity(y[right_ind]) if n_r > 0 else 0

#         # Weighted Gini index of child nodes
#         return (n_l / n) * gini_l + (n_r / n) * gini_r

#     def _information_gain(self,x_col,y,threshold):
#         parent_entropy = self._entropy(y)

#         left_ind,right_ind= self._split(x_col,threshold)
#         num = len(y)
#         n_l,n_r = len(left_ind),len(right_ind)
#         e_l,e_r = self._entropy(y[left_ind]),self._entropy(y[right_ind])
        
#         child_entropy = (n_l/num)*e_l + (n_r/num)*e_r

#         return parent_entropy-child_entropy

#     def _split(self,x_col,threshold):
#         left_ind = np.argwhere(x_col < threshold).flatten()
#         right_ind = np.argwhere(x_col>= threshold).flatten()
#         return left_ind,right_ind

#     def _entropy(self,y):
#         hist = np.bincount(y)
#         hist = hist/len(y)
#         return -np.sum([ele*np.log2(ele) for ele in hist if ele>0])

#     def _most_common_label(self,y):
#         if len(y)==0:
#             return None
#         counter = Counter(y)
#         return counter.most_common(1)[0][0]
    
#     def predict(self,x):
#         return [self.traverse_tree(ele,self.root) for ele in x]
    
#     def traverse_tree(self,x,node):
#         if node.is_leaf_node():
#             return node.value
        
#         if x[node.features]<node.threshold:
#             return self.traverse_tree(x,node.left)
#         return self.traverse_tree(x,node.right)

# # Example usage
# from sklearn import datasets
# from sklearn.model_selection import train_test_split

# data = datasets.load_breast_cancer()
# X, y = data.data, data.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234
# )

# clf = DT(max_depth=10)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# def accuracy(y_test, y_pred):
#     return np.sum(y_test == y_pred) / len(y_test)

# acc = accuracy(y_test, predictions)
# print("Accuracy:", acc)


import pandas as pd
import numpy as np
from collections import Counter

data = {
    "Income":[0,0,1,1,2,2],
    "Credit":[1,0,1,0,1,0],
    "Approved":[1,0,1,1,1,0]
}

df = pd.DataFrame(data)
print(df)

class Node:
    def __init__(self,threshold=None,features=None,left=None,right=None,value=None):
        self.threshold = threshold
        self.features = features
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self,max_depth=10,min_split=2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.root = None

    def fit(self,x,y):
        self.root = self._grow_tree(x,y)

    def _grow_tree(self,x,y,depth=0):
        n_samples,n_features =x.shape
        n_labels=len(np.unique(y))

        if depth>=self.max_depth or n_labels == 1 or n_samples<self.min_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_thresh, best_ind = self._best_split(x,y)

        left_ind,right_ind = self._split(x[best_ind],best_thresh)
        left = self._grow_tree(x[left_ind],y[left_ind],depth+1)
        right = self._grow_tree(x[right_ind],y[right_ind],depth+1)
        return Node(features=best_ind,threshold=best_thresh,left=left,right=right)

    def _best_split(self,x,y):
        best_gain = -1
        split_ind,split_thresh=-1,-1
        n_features = x.shape[1]

        for feat_ind in range(n_features):
            x_col = x[:,feat_ind]
            threshold=np.unique(x_col)

            for thr in threshold:
                gain = self._gini_impurity(x_col,y,thr)

                if gain>best_gain:
                    best_gain = gain
                    split_thresh = thr
                    split_ind = feat_ind

        return split_ind,split_thresh

    def _gini_impurity(self,x,y,thr):
        
        left_ind,right_ind = self._split(x,thr)
        num = len(y)
        n_l,n_r = len(left_ind),len(right_ind)

        gini_l = self._gini_index(y[left_ind])
        gini_r = self._gini_index(y[right_ind])

        return (n_l/num)*gini_l + (n_r/num)*gini_r
    
    def _gini_index(self,y):
        labels,counts=np.unique(y,return_counts=True)
        probability = counts/counts.sum()
        return 1-np.sum(probability**2)

    def _information_gain(self,x,y,threshold):
        parent_entropy = self._entropy(y)

        left_ind,right_ind = self._split(x,threshold)
        num = len(y)
        n_l,n_r = len(left_ind),len(right_ind)
        e_l,e_r = self._entropy(y[left_ind]),self._entropy(y[right_ind])

        child_entropy = (n_l/num)*e_l + (n_r/num)*e_r

        return parent_entropy-child_entropy

    def _split(self,x,threshold):
        left_ind = np.argwhere(x<threshold).flatten()
        right_ind = np.argwhere(x>=threshold).flatten()
        return left_ind, right_ind

    def _entropy(self,y):
        hist = np.bincount(y)
        hist = hist/len(y)
        return -np.sum([p*np.log2(p) for p in hist if p>0])
        

    def _most_common_label(self,y):
        if(len(y) == 0):
            return None
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self,x):
        return [self._traverse_tree(ele,self.root) for ele in x]

    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.features] < node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)

dt = DecisionTree()
df_values = df.values
x, y = df_values[:, :-1], df_values[:, -1]
dt.fit(x,y)

predicitons = dt.predict([[2,0]])

print(predicitons)