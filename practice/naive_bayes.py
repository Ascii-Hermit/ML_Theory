import numpy as np
from collections import defaultdict
import re

class NaiveBayesTextClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_totals = defaultdict(int)
        self.class_priors = {}
        self.vocabulary = set()
        
    def preprocess(self, text):
        # Basic text preprocessing: lowercasing, removing punctuation
        text = text.lower()
        text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (1-2 letters)
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()

    def fit(self, X, y):
        # Calculate priors and word frequencies for each class
        unique_classes = np.unique(y)
        for c in unique_classes:
            X_c = X[y == c]  # Texts belonging to class `c`
            self.class_priors[c] = len(X_c) / len(X)
            
            for text in X_c:
                words = self.preprocess(text)
                for word in words:
                    self.class_word_counts[c][word] += 1
                    self.vocabulary.add(word)
                self.class_totals[c] += len(words)
        print(self.class_word_counts)

    def predict(self, X):
        predictions = []
        for text in X:
            words = self.preprocess(text)
            class_scores = {}
            
            for c in self.class_priors:
                # Start with the log prior for each class
                log_prob = np.log(self.class_priors[c])
                for word in words:
                    word_count = self.class_word_counts[c].get(word, 0)
                    word_prob = (word_count + 1) / (self.class_totals[c] + len(self.vocabulary))
                    log_prob += np.log(word_prob)  # Add log of word probabilities
                
                class_scores[c] = log_prob
            
            # Select the class with the highest score
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions







# Sample data
X_train = np.array(["I love programming", "Python is amazing", "I dislike bugs", "Debugging is fun"])
y_train = np.array([1, 1, 0, 0])

# Training
nb_classifier = NaiveBayesTextClassifier()
nb_classifier.fit(X_train, y_train)

# Testing
X_test = np.array(["I love debugging", "I hate programming"])
predictions = nb_classifier.predict(X_test)
print("Predictions:", predictions)  # Outputs class labels for each test sentence
